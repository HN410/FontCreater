
import torch
import torch.nn as nn
from ..EfficientNet.model import *
from ..StyleGAN.network import *



class MyPSP(nn.Module):
    # 複数画像からフォントを構成するモデル
    def __init__(self, device):
        # chara_encoder ... どの文字かをエンコード
        # style_encoder ... 複数のフォントの組からスタイル情報をエンコード
        # style_gen ... エンコーダから得られた情報をもとにフォントを構成
        super().__init__()
        self.z_dim = 256 # エンコーダから渡される特徴量の個数
        blocks_args, global_params = get_model_params('efficientnet-b0', {})
        self.chara_encoder = EfficientNetEncoder(blocks_args, global_params)
        self.chara_encoder._change_in_channels(1)
        self.style_encoder = EfficientNetEncoder(blocks_args, global_params)
        self.style_encoder._change_in_channels(1)
        gen_settings = get_setting_json()
        self.style_gen = Generator(gen_settings["network"])
        self.alpha = torch.ones((1)).to(device)
    
    def forward(self, chara_images,  style_pairs):
        # chara_image ... 変換したい文字のMSゴシック体の画像
        #   [B, 1, 256, 256]
        # style_pairs ... MSゴシック体の文字と、その文字に対応する変換先のフォントの文字の画像のペアのテンソル
        #   [B, pair_n, 2, 1, 256, 256]
        batch_n = chara_images.size()[0]
        pair_n = style_pairs.size()[1]

        chara_images = self.chara_encoder(chara_images)
        
        # ペアの差分をとる [B, pair_n, 1, 256, 256]
        style_pairs = style_pairs[:, :, 1] -  style_pairs[:, :, 0]
        # 文字ごとにencoderにかけ、その特徴量を総和する [B, 256*4, 1, 1]
        style_pairs = [self.style_encoder(style_pairs[:, i]) for i in range(pair_n)]
        print(style_pairs[0].size())
        style_pairs = torch.stack(style_pairs).sum(0)
        print(chara_images[:, :self.z_dim*2].size())
        print(style_pairs[:, self.z_dim*2:].size())

        self.style_gen(chara_images[:, :self.z_dim*2], style_pairs[:, self.z_dim*2:], self.alpha)
