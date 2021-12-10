from torch import nn
class Map2Style(nn.Module):
    # Encoderの出力から、Generator用に3つのconvolutionを通してwを出力する
    FINAL_CHANNEL_N = 256
    CONV_PARAMS = [(1, 4, 2), (1, 6, 4), (1, 10, 8)] # 畳み込みのパラメータ。左から1/2, 1/4, 1/8にする。それぞれ(padding, kernel, stride)
    MAX_IDX = 4
    SCALE_LIST = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]] # 各畳み込みでCONV_PARAMSの何番目を使うか
    CHARACTER_OUTC = [[256, 512, 512], [256, 512, 1024]] # 文字用の各畳み込み層のチャンネル出力数

    # idxは特徴マップのサイズが8, 16, 32, 64のうちどれかに対応
    def __init__(self, idx, channelN, useBN = False):
        super(Map2Style, self).__init__()
        assert 0 <= idx < Map2Style.MAX_IDX
        self.__setConvs__(idx, channelN, useBN)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.useBN = useBN
    
    def __setConvs__(self, idx, inChannel, useBN):
        # idx >= 2　かどうかで文字用かスタイル用か区別
        self.convs = []
        self.BNs = []
        outC = Map2Style.FINAL_CHANNEL_N
        for i, scale in enumerate(Map2Style.SCALE_LIST[idx]):
            inC = outC
            if(i == 0):
                inC = inChannel
                if(idx >= Map2Style.MAX_IDX // 2):
                    outC = outC // 2
            if(i == 1 and idx >= Map2Style.MAX_IDX // 2): #スタイル用は基本出力チャンネル数256
                outC = Map2Style.FINAL_CHANNEL_N
            if(idx < Map2Style.MAX_IDX // 2): #文字用
                outC = self.CHARACTER_OUTC[idx][i]
            convParams = Map2Style.CONV_PARAMS[scale]
            conv = nn.Conv2d(inC, outC, convParams[1],  convParams[2], convParams[0])
            self.convs.append(conv)

            # BNを挟む
            if(useBN):
                self.BNs.append(nn.BatchNorm2d(outC))
        self.convs = nn.ModuleList(self.convs)
        if(useBN):
            self.BNs = nn.ModuleList(self.BNs)
    
    def forward(self, input):
        for i, conv in enumerate(self.convs):
            input = conv(input)
            if(self.useBN):
                input = self.BNs[i](input)
            input = self.activation(input)
        return input