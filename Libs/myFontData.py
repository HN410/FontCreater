import random
from .myFontLib import *
import torch.utils.data as data
class FontGeneratorDataset(data.Dataset):
    # ゴシック体と各フォントのペア画像の組を出力するデータセット

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
    
    def resetSampleN(self):
        self.sampleN = random.randint(self.imageN[0], self.imageN[1])