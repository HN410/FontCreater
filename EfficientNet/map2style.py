from torch import nn
class Map2Style(nn.Module):
    # Encoderの出力から、Generator用に3つのconvolutionを通してwを出力する
    FINAL_CHANNEL_N = 256
    CONV_PARAMS = [(1, 4, 2), (1, 6, 4), (1, 10, 8)] # 畳み込みのパラメータ。左から1/2, 1/4, 1/8にする。それぞれ(padding, kernel, stride)
    MAX_IDX = 4
    SCALE_LIST = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]] # 各畳み込みでCONV_PARAMSの何番目を使うか

    # idxは特徴マップのサイズが8, 16, 32, 64のうちどれかに対応
    def __init__(self, idx, channelN):
        super(Map2Style, self).__init__()
        assert 0 <= idx < Map2Style.MAX_IDX
        self.__setConvs__(idx, channelN)
    
    def __setConvs__(self, idx, inChannel):
        self.convs = []
        outC = Map2Style.FINAL_CHANNEL_N
        for i, scale in enumerate(Map2Style.SCALE_LIST[idx]):
            inC = outC
            if(i == 0):
                inC = inChannel
                if(idx >= Map2Style.MAX_IDX // 2):
                    outC = outC // 2
            if(i == 1 and idx >= Map2Style.MAX_IDX // 2):
                outC = Map2Style.FINAL_CHANNEL_N
            convParams = Map2Style.CONV_PARAMS[scale]
            conv = nn.Conv2d(inC, outC, convParams[1],  convParams[2], convParams[0])
            self.convs.append(conv)
        self.convs = nn.ModuleList(self.convs)
    
    def forward(self, input):
        for conv in self.convs:
            input = conv(input)
        return input