import random
import torch
from .myFontLib import *
import torch.utils.data as data
class FontGeneratorDataset(data.Dataset):
    # ゴシック体と各フォントのペア画像の組を出力するデータセット

    # 平均、分散を計算するときにサンプリングする文字数、フォント数
    MEANVAR_CHARA_N = 40
    MEANVAR_FONT_N = 70

    def __init__(self, fontTools: FontTools, compatibleDict: dict, imageN : list, useTensor=True):
        #  fontTools ... FontTools
        #  compatibleDict ... 各フォントごとに対応している文字のリストを紐づけたディクショナリ
        #  imageN ... ペア画像を出力する数の範囲(要素は２つ)
        #             (例) [3, 6] ... 3~6個の中から一様分布で決定される
        #                  [4, 4] ...4個で固定
        self.fontTools = fontTools
        self.fontList = FontTools.getFontPathList()
        self.compatibleDict = compatibleDict
        self.imageN = imageN
        self.resetSampleN()
        self.useTensor = useTensor
    def __len__(self):
        return len(self.fontList)
    
    def __getitem__(self, index):
        charaChooser = CharacterChooser(self.fontTools, self.fontList[index],
                 self.compatibleDict[self.fontList[index]], useTensor=self.useTensor)
        sampleN = self.sampleN
        return charaChooser.getSampledImagePair(sampleN)

    @classmethod
    def getCharaImagesMeanVar(cls, compatibleData):
        # フォント画像の平均、分散を得る

        fontDataSet = FontGeneratorDataset(FontTools(), compatibleData, [cls.MEANVAR_CHARA_N, cls.MEANVAR_CHARA_N], useTensor=True)
        fontDataSet = iter(fontDataSet)
        # [FONT_N, CHARA_N, 2, 3, 256, 256]
        data = torch.cat([torch.cat([torch.cat(j, 0) for j in list(fontDataSet.__next__())]) for i in range(cls.MEANVAR_FONT_N)])
        mean = torch.mean(data).item()
        var = torch.var(data).item()
        return mean, var
    
    def resetSampleN(self):
        self.sampleN = random.randint(self.imageN[0], self.imageN[1])