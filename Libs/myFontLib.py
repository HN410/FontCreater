import os
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import matplotlib.pyplot as plt
import pickle
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
import random


class FontCheckImageProducer():
    # フォントを確認する用の画像を作るクラス
    jisChineseCharas = ["JIS_first.txt", "JIS_second.txt"]
    fontDirs =  ["Fonts"]
    maxCharas = 180 # チェック時に見る最大文字数
    charasHorizontalN = 30  # 横に並べる文字数    
    imageUnitSize = (600, 150)
    imageSize = (imageUnitSize[0]*2, imageUnitSize[1]*2)
    fontSize = 20
    
    def getFontPathList():
        dirs = FontCheckImageProducer.fontDirs

        l = []
        for d in dirs:
            for parent, _, filenames in os.walk(d):
                for name in filenames:
                    l.append(os.path.join(parent, name))
        return l

    def getJISList():
        ans = []
        for path in FontCheckImageProducer.jisChineseCharas:
            data = ""
            with open(path, "r", encoding='UTF-8') as f:
                data = f.read()
                data = "".join(data.split("\n"))
            ans.append(data)
        return ans
    def getFontCheckStrings():
        fontCheckStrings = ["0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", 
                        "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲン", 
                        ]
        fontCheckStrings += FontCheckImageProducer.getJISList()
        return fontCheckStrings
    
    def __init__(self):
        # 文字数が多いカテゴリはランダムにサンプリング
        self.fontPathList = FontCheckImageProducer.getFontPathList()
        self.fontCheckStrings = FontCheckImageProducer.getFontCheckStrings()
        for i, string in enumerate(self.fontCheckStrings):
            if(len(string) > FontCheckImageProducer.maxCharas):
                sampled = random.sample(list(string), FontCheckImageProducer.maxCharas)
                string = "".join(sampled)
                self.fontCheckStrings[i] = string
            # 表示しやすいようにこの時点で整形
            horiN = FontCheckImageProducer.charasHorizontalN
            if(len(string) > horiN):
                strList = [string[horiN*i:horiN*(i+1)] for i in range(len(string) // horiN + 1)]
                string = "\n".join(strList)
                self.fontCheckStrings[i] = string


    def getFontImage(self, ind):
        font_path = self.fontPathList[ind]
        font = PIL.ImageFont.truetype(font_path, FontCheckImageProducer.fontSize)

        image = PIL.Image.new('RGBA', FontCheckImageProducer.imageSize, 'white')
        draw = PIL.ImageDraw.Draw(image)
        for i, string in enumerate(self.fontCheckStrings):
            draw.text(((FontCheckImageProducer.imageUnitSize[0]) * (i % 2), (FontCheckImageProducer.imageUnitSize[1]) * (i // 2)),
                    string,
                    font=font,
                    fill='black')

        return image

