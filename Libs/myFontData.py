import random
import torch
from .myFontLib import *
import torch.utils.data as data
class FontGeneratorDataset(data.Dataset):
    # ゴシック体と各フォントのペア画像の組を出力するデータセット

    # 平均、分散を計算するときにサンプリングする文字数、フォント数
    MEANVAR_CHARA_N = 40
    MEANVAR_FONT_N = 70

    IMAGE_MEAN = 0.8893
    IMAGE_VAR = 0.0966

    def __init__(self, fontTools: FontTools, compatibleDict: dict, imageN : list,\
         useTensor=True, startInd = 0, indN = None, isForValid = None):
        #  fontTools ... FontTools
        #  compatibleDict ... 各フォントごとに対応している文字のリストを紐づけたディクショナリ
        #  imageN ... ペア画像を出力する数の範囲(要素は２つ)
        #             (例) [3, 6] ... 3~6個の中から一様分布で決定される
        #                  [4, 4] ...4個で固定
        #  startInd ... fontListのうち、このインデックス以降のフォントのみを使う
        #  indN ...startIndからindN個のフォントのみを使う。NoneならstartInd以降すべて
        #  isForValid ... validationなどで、常に固定したデータで出力をしたいときに使う
        # 　　getInputListForVで取得したディクショナリをここに入れればよい。
        self.fontTools = fontTools
        self.fontList = FontTools.getFontPathList()
        self.compatibleDict = compatibleDict
        self.imageN = imageN
        self.resetSampleN()
        self.useTensor = useTensor
        self.startInd = startInd
        if(indN is None):
            self.indN = len(self.fontList) - startInd
        else:
            self.indN = indN

        self.transform = transforms.Compose([
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_VAR)
        ])
        if(isForValid is not None):
            self.isForValid = True
            self.fixedInput = isForValid
        else:
            self.isForValid = False

    def __len__(self):
        return self.indN
    
    def __getitem__(self, index):
        # 形式は変換用の画像の組と教師用データのテンソルのリスト
        # 変換用画像 idx:0 [1, 256, 256]の変換元画像
        #           idx:1 [1, 256, 256]の変換後画像
        # 教師用データ [imageN-1, 2, 1, 256, 256]のゴシック、変換後フォントの文字の画像のペアのテンソル
        

        # まず、入力されたindexを補正
        index += self.startInd

        imageList = []

        charaChooser = CharacterChooser(self.fontTools, self.fontList[index],
                self.compatibleDict[self.fontList[index]], useTensor=self.useTensor)
        if(self.isForValid):
            imageList = charaChooser.getImageFromSampleList(self.fixedInput[index], self.transform)
        else:
            sampleN = self.sampleN
            imageList = charaChooser.getSampledImagePair(sampleN, self.transform)

        convertedPair = imageList[0]
        teachers = torch.stack([torch.stack(i, 0) for i in imageList[1:]], 0)
        
        return [convertedPair, teachers]
    
    def getInputListForV(self):
        # validationように常に固定された入力が出るよう、このデータセットに設定するディクショナリを作る
        # 形式は、フォントのインデックスをキーとする文字のリストのディクショナリ
        sampleN = random.randint(self.imageN[0], self.imageN[1])
        ans = {}
        for i in range(self.__len__()):
            charaChooser = CharacterChooser(self.fontTools, self.fontList[self.startInd+ i],
                 self.compatibleDict[self.fontList[self.startInd + i]])
            ans[i] = charaChooser.sample(sampleN)
        return ans

    def getJapaneseFontIndices(self):
        # 日本語の文字を含むフォントに対応するインデックスのリストを返す
        index = self.startInd
        ans = []
        for i in range(self.indN):
            font = self.fontList[index]
            compatibleList = self.compatibleDict[font]
            for j in range(2, 5):
                if compatibleList[j]:
                    ans.append(i)
                    break
            index+=1
            i+=1
        return ans 


    @classmethod
    def getCharaImagesMeanVar(cls, compatibleData, isMinus = False):
        # フォント画像の平均、分散を得る

        fontDataSet = FontGeneratorDataset(FontTools(), compatibleData, [cls.MEANVAR_CHARA_N, cls.MEANVAR_CHARA_N], useTensor=True)
        fontDataSet = iter(fontDataSet)
        # [FONT_N, CHARA_N, 2, 1, 256, 256]
        data = torch.cat([torch.cat([torch.cat(j, 0) for j in list(fontDataSet.__next__())]) for i in range(cls.MEANVAR_FONT_N)])
        mean = torch.mean(data).item()
        var = torch.var(data).item()
        return mean, var
    
    def resetSampleN(self):
        self.sampleN = random.randint(self.imageN[0], self.imageN[1])
    

class MyPSPBatchSampler(torch.utils.data.sampler.BatchSampler):
    # MyPSP用のBatchSampler
    def __init__(self, batchSize, fontGeneratorDataset: FontGeneratorDataset, japaneseRate = 0):
        self.fontGeneratorDataset = fontGeneratorDataset
        self.len = len(fontGeneratorDataset)
        self.batchSize = batchSize
        if( 0 < japaneseRate <= 1):
            self.japaneseRate = japaneseRate
        else:
            self.japaneseRate = 0
        
    def __iter__(self):
        self.count = self.batchSize
        self.indicesList = random.sample(list(range(self.len)), self.len)
        if(self.japaneseRate > 0):
            self.japaneseIndicesList = random.choices(self.fontGeneratorDataset.getJapaneseFontIndices(), k=self.len)
        while self.count <= self.len:
            self.fontGeneratorDataset.resetSampleN()
            if(random.random() < self.japaneseRate):
                yield(self.japaneseIndicesList[self.count-self.batchSize: self.count])
            else:
                yield(self.indicesList[self.count-self.batchSize: self.count])
            self.count += self.batchSize
    
    def __len__(self):
        return self.len // self.batchSize

        