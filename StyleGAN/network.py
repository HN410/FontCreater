import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os 
import json

SETTING_JSON_PATH = "./settings.json"

class PixelNormalizationLayer(nn.Module):
    # チャンネル方向に正規化する
    def __init__(self, settings):
        super().__init__()
        self.epsilon = settings["epsilon"]

    def forward(self, x):
        # x is [B, C, H, W]
        if x is None:
            return x
        x2 = x ** 2

        length_inv = torch.rsqrt(x2.mean(1, keepdim=True) + self.epsilon)

        return x * length_inv


class MinibatchStdConcatLayer(nn.Module):
    def __init__(self, settings):
        super().__init__()

        self.num_concat = settings["std_concat"]["num_concat"]
        self.group_size = settings["std_concat"]["group_size"]
        self.use_variance = settings["std_concat"]["use_variance"]  # else use variance
        self.epsilon = settings["epsilon"]

    def forward(self, x):
        if self.num_concat == 0:
            return x

        group_size = self.group_size
        # x is [B, C, H, W]
        size = x.size()
        assert(size[0] % group_size == 0)
        M = size[0]//group_size

        x32 = x.to(torch.float32)

        y = x32.view(group_size, M, -1)  # [group_size, M, -1]
        mean = y.mean(0, keepdim=True)  # [1, M, -1]
        y = ((y - mean)**2).mean(0)  # [M, -1]
        if not self.use_variance:
            y = (y + self.epsilon).sqrt()
        y = y.mean(1)  # [M]
        y = y.repeat(group_size, 1)  # [group_size, M]
        y = y.view(-1, 1, 1, 1)
        y1 = y.expand([size[0], 1, size[2], size[3]])
        y1 = y1.to(x.dtype)

        if self.num_concat == 1:
            return torch.cat([x, y1], 1)

        # self.num_concat == 2
        y = x32.view(M, group_size, -1)  # [M, group_size, -1]
        mean = y.mean(1, keepdim=True)  # [M, 1, -1]
        y = ((y - mean) ** 2).mean(1)  # [M, -1]
        if self.use_variance:
            y = (y + 1e-8).sqrt()
        y = y.mean(1, keepdim=True)  # [M, 1]
        y = y.repeat(1, group_size)  # [M, group_size]
        y = y.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        y2 = y.expand([size[0], 1, size[2], size[3]])
        y2 = y2.to(x.dtype)

        return torch.cat([x, y1, y2], 1)


class Blur3x3(nn.Module):
    def __init__(self):
        super().__init__()

        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[None, :] * f[:, None]
        f /= np.sum(f)
        f = f.reshape([1, 1, 3, 3])
        self.register_buffer("filter", torch.from_numpy(f))

    def forward(self, x):
        ch = x.size(1)
        return F.conv2d(x, self.filter.expand(ch, -1, -1, -1), padding=1, groups=ch)


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        # Moduleに紐づく定数を設定するときに使う
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        scaled_weight = self.weight * self.scale
        return F.conv2d(x, scaled_weight, self.bias, self.stride, self.padding)


class WSConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, gain=np.sqrt(2)):
        super().__init__()
        weight = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        init.normal_(weight)
        self.weight = nn.Parameter(weight)
        scale = gain / np.sqrt(in_channels * kernel_size * kernel_size)
        self.register_buffer("scale", torch.tensor(scale))

        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        scaled_weight = self.weight * self.scale
        return F.conv_transpose2d(x, scaled_weight, self.bias, self.stride, self.padding)


class AdaIN(nn.Module):
    def __init__(self, dim, w_dim):
        super().__init__()
        self.dim = dim
        self.epsilon = 1e-8
        self.scale_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)
        self.bias_transform = WSConv2d(w_dim, dim, 1, 1, 0, gain=1)

    def forward(self, x, w):
        x = F.instance_norm(x, eps=self.epsilon)

        # scale
        scale = self.scale_transform(w)
        bias = self.bias_transform(w)

        return scale * x + bias


class NoiseLayer(nn.Module):
    def __init__(self, dim, size):
        super().__init__()

        self.fixed = False

        self.size = size
        self.register_buffer("fixed_noise", torch.randn([1, 1, size, size]))

        self.noise_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        batch_size = x.size()[0]
        if self.fixed:
            noise = self.fixed_noise.expand(batch_size, -1, -1, -1)
        else:
            noise = torch.randn([batch_size, 1, self.size, self.size], dtype=x.dtype, device=x.device)
        return x + noise * self.noise_scale




class SynthFirstBlock(nn.Module):
    def __init__(self, start_dim, output_dim, w_dim, base_image_init, use_noise):
        super().__init__()

        self.base_image = nn.Parameter(torch.empty(1, start_dim, 4, 4))
        if base_image_init == "zeros":
            nn.init.zeros_(self.base_image)
        elif base_image_init == "ones":
            nn.init.ones_(self.base_image)
        elif base_image_init == "zero_normal":
            nn.init.normal_(self.base_image, 0, 1)
        elif base_image_init == "one_normal":
            nn.init.normal_(self.base_image, 1, 1)
        else:
            print(f"Invalid base_image_init: {base_image_init}")
            exit(1)

        self.conv = WSConv2d(start_dim, output_dim, 3, 1, 1)

        self.noise1 = NoiseLayer(start_dim, 4)
        self.noise2 = NoiseLayer(output_dim, 4)
        if not use_noise:
            self.noise1.noise_scale.zeros_()
            self.noise1.fixed = True
            self.noise2.noise_scale.zeros_()
            self.noise2.fixed = True

        self.adain1 = AdaIN(start_dim, w_dim)
        self.adain2 = AdaIN(output_dim, w_dim)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, w1, w2):

        batch_size = w1.size()[0]

        x = self.base_image.expand(batch_size, -1, -1, -1)
        x = self.noise1(x)
        x = self.activation(x)
        x = self.adain1(x, w1)

        x = self.conv(x)
        x = self.noise2(x)
        x = self.activation(x)
        x = self.adain2(x, w2)

        return x


class SynthBlock(nn.Module):
    def __init__(self, input_dim, output_dim, output_size, w_dim, upsample_mode, use_blur, use_noise):
        super().__init__()

        self.conv1 = WSConv2d(input_dim, output_dim, 3, 1, 1)
        self.conv2 = WSConv2d(output_dim, output_dim, 3, 1, 1)
        if use_blur:
            self.blur = Blur3x3()
        else:
            self.blur = None

        self.noise1 = NoiseLayer(output_dim, output_size)
        self.noise2 = NoiseLayer(output_dim, output_size)
        if not use_noise:
            self.noise1.noise_scale.zeros_()
            self.noise1.fixed = True
            self.noise2.noise_scale.zeros_()
            self.noise2.fixed = True

        self.adain1 = AdaIN(output_dim, w_dim)
        self.adain2 = AdaIN(output_dim, w_dim)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.upsample_mode = upsample_mode

    def forward(self, x, w1, w2):

        x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        x = self.conv1(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.noise1(x)
        x = self.activation(x)
        x = self.adain1(x, w1)

        x = self.conv2(x)
        x = self.noise2(x)
        x = self.activation(x)
        x = self.adain2(x, w2)

        return x


class SynthesisModule(nn.Module):
    def __init__(self, settings, make_blocks = True):
        super().__init__()

        self.w_dim = settings["w_dim"]
        self.upsample_mode = settings["upsample_mode"]
        use_blur = settings["use_blur"]
        use_noise = settings["use_noise"]
        base_image_init = settings["base_image_init"]

        if make_blocks:
            self.blocks = nn.ModuleList([
                SynthFirstBlock(256, 256, self.w_dim, base_image_init, use_noise),
                SynthBlock(256, 256, 8, self.w_dim, self.upsample_mode, use_blur, use_noise),
                SynthBlock(256, 256, 16, self.w_dim, self.upsample_mode, use_blur, use_noise),
                SynthBlock(256, 128, 32, self.w_dim, self.upsample_mode, use_blur, use_noise),
                SynthBlock(128, 64, 64, self.w_dim, self.upsample_mode, use_blur, use_noise),
                SynthBlock(64, 32, 128, self.w_dim, self.upsample_mode, use_blur, use_noise),
                SynthBlock(32, 16, 256, self.w_dim, self.upsample_mode, use_blur, use_noise)
            ])
            self.to_monos = nn.ModuleList([
                WSConv2d(256, 1, 1, 1, 0, gain=1),
                WSConv2d(256, 1, 1, 1, 0, gain=1),
                WSConv2d(256, 1, 1, 1, 0, gain=1),
                WSConv2d(128, 1, 1, 1, 0, gain=1),
                WSConv2d(64, 1, 1, 1, 0, gain=1),
                WSConv2d(32, 1, 1, 1, 0, gain=1),
                WSConv2d(16, 1, 1, 1, 0, gain=1)
            ])
        

        # self.to_rgbs = nn.ModuleList([
        #     WSConv2d(256, 3, 1, 1, 0, gain=1),
        #     WSConv2d(256, 3, 1, 1, 0, gain=1),
        #     WSConv2d(256, 3, 1, 1, 0, gain=1),
        #     WSConv2d(128, 3, 1, 1, 0, gain=1),
        #     WSConv2d(64, 3, 1, 1, 0, gain=1),
        #     WSConv2d(32, 3, 1, 1, 0, gain=1),
        #     WSConv2d(16, 3, 1, 1, 0, gain=1)
        # ])


        self.chara_training_convs = nn.ModuleList([
            nn.Conv2d(256, 128, 3, 1, 1), 
            nn.Conv2d(128, 1, 3, 1, 1)
        ])

        self.level = 7
        self.for_chara_training = False

        # self.register_buffer("level", torch.tensor(1, dtype=torch.int32))
    def set_level(self, level: int):
        assert 0 < level <= 7
        self.level = level
    def set_for_chara_training(self, b):
        self.for_chara_training = b
        if b:
            self.level = min(self.level, 3)

    def set_noise_fixed(self, fixed):
        for module in self.modules():
            if isinstance(module, NoiseLayer):
                module.fixed = fixed
    
    def chara_training_layer(self, x):
        for i in range(2):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x = self.chara_training_convs[i](x)
            x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = F.interpolate(x, size=(256, 256), mode = "bicubic")
        return x

    def forward(self, w, alpha):
        # w is [batch_size. level*2, w_dim, 1, 1]
        # levelは7の固定
        level = self.level

        x = self.blocks[0](w[:, 0], w[:, 1])

        if level == 1:
            x = self.to_monos[0](x)
            x = F.interpolate(x, scale_factor=2**(7-level), mode = "bilinear")
            return x

        for i in range(1, level-1):
            x = self.blocks[i](x, w[:, i*2], w[:, i*2+1])

        x2 = x
        x2 = self.blocks[level-1](x2, w[:, level*2-2], w[:, level*2-1])
        if self.for_chara_training and self.level == 3:
            return self.chara_training_layer(x2)


        x2 = self.to_monos[level-1](x2)

        if alpha.item() == 1:
            x = x2
        else:
            x1 = self.to_monos[level-2](x)
            x1 = F.interpolate(x1, scale_factor=2, mode=self.upsample_mode)
            x = torch.lerp(x1, x2, alpha.item())
        
        if level < 7:
            x = F.interpolate(x, scale_factor=2**(7-level), mode = "bilinear")
        return x

    def write_histogram(self, writer, step):
        for lv in range(self.level):
            block = self.blocks[lv]
            for name, param in block.named_parameters():
                writer.add_histogram(f"g_synth_block{lv}/{name}", param.cpu().data.numpy(), step)

        for name, param in self.to_rgbs.named_parameters():
            writer.add_histogram(f"g_synth_block.torgb/{name}", param.cpu().data.numpy(), step)

class SynthesisModule2(SynthesisModule):
    # myPSP ver2用のsyntethis. forwardの入力が特徴量マップとwになる
    def __init__(self, settings):
        super().__init__(settings, make_blocks=False)
        use_blur = settings["use_blur"]
        use_noise = settings["use_noise"]
        self.first_conv = WSConv2d(320, 256, 3, 1, 1, gain=1)
        self.blocks = nn.ModuleList([
            SynthBlock(256, 128, 32, self.w_dim, self.upsample_mode, use_blur, use_noise),
            SynthBlock(128, 64, 64, self.w_dim, self.upsample_mode, use_blur, use_noise),
            SynthBlock(64, 32, 128, self.w_dim, self.upsample_mode, use_blur, use_noise),
            SynthBlock(32, 16, 256, self.w_dim, self.upsample_mode, use_blur, use_noise)
        ])
        self.to_monos = nn.ModuleList([
            WSConv2d(320, 1, 1, 1, 0, gain=1),
            WSConv2d(256, 1, 1, 1, 0, gain=1),
            WSConv2d(128, 1, 1, 1, 0, gain=1),
            WSConv2d(64, 1, 1, 1, 0, gain=1),
            WSConv2d(32, 1, 1, 1, 0, gain=1),
            WSConv2d(16, 1, 1, 1, 0, gain=1)
        ])

    def set_level(self, level: int):
        assert 0 < level <= 6
        self.level = level
    def set_for_chara_training(self, b):
        self.for_chara_training = b
        if b:
            self.level = min(self.level, 2)

    def forward(self, chara_z, w, alpha):
        # w is [batch_size. 8, w_dim, 1, 1]
        # chara_z is [batch_size, 320, 8, 8]
        level = self.level

        x = chara_z

        if level == 1:
            x = self.to_monos[0](x)
            x = F.interpolate(x, size=(256, 256), mode = "bilinear")
            return x
        
        x = F.interpolate(320, scale_factor=2, mode="bilinear")
        x = self.first_conv(x)


        for i in range(0, level-3):
            x = self.blocks[i](x, w[:, i*2], w[:, i*2+1])
        x2 = x
        x2= self.blocks[level-2](x2, w[:, i*2], w[:, i*2+1])

        if self.for_chara_training and self.level == 3:
            return self.chara_training_layer(x2)


        x2 = self.to_monos[level-1](x2)

        if alpha.item() == 1:
            x = x2
        else:
            x1 = self.to_monos[level-2](x)
            x1 = F.interpolate(x1, scale_factor=2, mode=self.upsample_mode)
            x = torch.lerp(x1, x2, alpha.item())
        
        if level < 6:
            x = F.interpolate(x, size = (256, 256), mode = "bilinear")
        return x


class Generator(nn.Module):
    def __init__(self, settings, for_chara_training = False, ver = 1):
        super().__init__()
        if ver == 1:
            self.synthesis_module = SynthesisModule(settings)
        elif ver == 2:
            self.synthesis_module = SynthesisModule2(settings)
        self.style_mixing_prob = settings["style_mixing_prob"]

        # Truncation trick
        self.register_buffer("w_average", torch.zeros(1, settings["z_dim"], 1, 1))
        self.w_average_beta = 0.995
        self.z_dim = settings["z_dim"]
        self.z_kind = 2 # Encoderからz_dim個の特徴量が何組ずつ流れ込むか
        self.trunc_w_layers = 8
        self.trunc_w_psi = 0.8
        self.latent_normalization = PixelNormalizationLayer(settings) if settings["normalize_latents"] else None
        self.for_chara_training = for_chara_training # 文字のエンコードデコードのみを訓練
        self.ver = ver
 

    def set_level(self, level):
        self.synthesis_module.set_level(level)

    def set_for_chara_training(self, b):
        self.for_chara_training = b
        self.synthesis_module.set_for_chara_training(b)

    def forward(self, chara_z, style_z, alpha):
        batch_size = chara_z.size()[0]
        level = self.synthesis_module.level

        if self.latent_normalization is not None:
            if(self.ver == 1):
                chara_z = self.latent_normalization(chara_z)
            style_z = self.latent_normalization(style_z)
        # chara_z[Batch, z_dim * 6, 1, 1]
        # style_z[Batch, z_dim * 2, 1, 1]
        z = self.expand_to_w(chara_z, style_z)
        # z becomes
        # → [B, 7*2, 256, 1, 1]


        # update w_average
        # if self.training:
        #     self.w_average = torch.lerp(z.mean(0, keepdim=True).detach(), self.w_average, self.w_average_beta)


        # style mixing
        # if self.training and level >= 2:
        #     z_mix = torch.randn_like(z)
        #     w_mix = self.latent_transform(z_mix)
        #     for batch_index in range(batch_size):
        #         if np.random.uniform(0, 1) < self.style_mixing_prob:
        #             cross_point = np.random.randint(1, level*2)
        #             z[batch_index, cross_point:] = w_mix[batch_index]

        # Truncation trick
        # if not self.training:
        #     w[:, self.trunc_w_layers:] = torch.lerp(self.w_average.clone(),
        #                                             w[:, self.trunc_w_layers:].clone(),
        #                                             self.trunc_w_psi).clone()
        fakes = None
        if self.ver == 1:
            fakes = self.synthesis_module(z, alpha)
        elif self.ver == 2:
            fakes = self.synthesis_module(chara_z, z, alpha)
            

        return fakes

    def write_histogram(self, writer, step):
        # for name, param in self.latent_transform.named_parameters():
        #     writer.add_histogram(f"g_lt/{name}", param.cpu().data.numpy(), step)
        self.synthesis_module.write_histogram(writer, step)
        writer.add_histogram("w_average", self.w_average.cpu().data.numpy(), step)

    def expand_to_w(self, chara_z, style_z):
        # zは[Batch, z_dim * (6+2), 1, 1]
        # これをstyle_zのみ拡大
        batch_size = chara_z.size()[0]
        if self.ver != 2:
            chara_z = [chara_z[:,self.z_dim * i : self.z_dim*(i+1) ] for i in range(chara_z.size()[1]//self.z_dim)]
            chara_z = [e.view(batch_size, 1, -1, 1, 1) for e in chara_z]
            chara_z = torch.cat(chara_z, 1)
        if style_z is None:
            return chara_z
        style_z =[style_z[:, self.z_dim*i : self.z_dim * (i+1)] for i in range(self.z_kind)] 
        z23 = [style_z[i].view(batch_size, 1, -1, 1, 1).expand(batch_size, 4, self.z_dim, 1, 1) for i in range(self.z_kind)]

        if(self.ver == 2):
            return torch.cat(z23, 1)

        return torch.cat((chara_z, z23[0], z23[1]), 1)


class DBlock(nn.Module):
    def __init__(self, inpit_dim, output_dim, use_blur):
        super().__init__()

        self.conv1 = WSConv2d(inpit_dim, output_dim, 3, 1, 1)
        self.conv2 = WSConv2d(output_dim, output_dim, 3, 1, 1)
        if use_blur:
            self.blur = Blur3x3()
        else:
            self.blur = None
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        if self.blur is not None:
            x = self.blur(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class DLastBlock(nn.Module):
    def __init__(self, input_dim, label_size):
        super().__init__()

        self.conv1 = WSConv2d(input_dim, input_dim, 3, 1, 1)
        self.conv2 = WSConv2d(input_dim, input_dim, 4, 1, 0)
        self.conv3 = WSConv2d(input_dim, label_size, 1, 1, 0, gain=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, settings, label_size):
        super().__init__()

        use_blur = settings["use_blur"]
        self.downsample_mode = settings["upsample_mode"]

        self.from_monos = nn.ModuleList([
            WSConv2d(1, 16, 1, 1, 0),
            WSConv2d(1, 32, 1, 1, 0),
            WSConv2d(1, 64, 1, 1, 0),
            WSConv2d(1, 128, 1, 1, 0),
            WSConv2d(1, 256, 1, 1, 0),
            WSConv2d(1, 256, 1, 1, 0),
            WSConv2d(1, 256, 1, 1, 0)
        ])

        self.blocks = nn.ModuleList([
            DBlock(16, 32, use_blur),
            DBlock(32, 64, use_blur),
            DBlock(64, 128, use_blur),
            DBlock(128, 256, use_blur),
            DBlock(256, 256, use_blur),
            DBlock(256, 256, use_blur),
            DLastBlock(256, self.label_size)
        ])

        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.register_buffer("level", torch.tensor(1, dtype=torch.int32))

    def set_level(self, level):
        self.level.fill_(level)

    def forward(self, x, alpha):
        level = self.level.item()

        if level == 1:
            x = self.from_monos[-1](x)
            x = self.activation(x)
            x = self.blocks[-1](x)
        else:
            x2 = self.from_monos[-level](x)
            x2 = self.activation(x2)
            x2 = self.blocks[-level](x2)

            if alpha == 1:
                x = x2
            else:
                x1 = F.interpolate(x, scale_factor=0.5, mode=self.downsample_mode)
                x1 = self.from_monos[-level+1](x1)
                x1 = self.activation(x1)

                x = torch.lerp(x1, x2, alpha)

            for l in range(1, level):
                x = self.blocks[-level+l](x)

        return x.view([-1, 1])

def get_setting_json():
    dirname = os.path.dirname(__file__)
    setting_path = os.path.join(dirname, SETTING_JSON_PATH)
    settings = None
    with open(setting_path) as fp:
        settings = json.load(fp)
    return settings
