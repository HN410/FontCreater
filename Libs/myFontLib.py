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
            path = os.path.dirname(__file__) + "\\" + path
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

class FontChecker():
    # フォントを表示しながら、どれに対応するかを記録していくGUI
    tagListIndex = ["英数字", "記号", "かな", "JIS第一水準", "JIS第二水準", "特殊"]
    checkListColumns = 3
    def __init__(self, fontCheckImageProducer):
        self.fontN = 0
        self.nowInd = -1
        self.fontList =  FontCheckImageProducer.getFontPathList()
        self.fontN = len(self.fontList)
        self.data = { i: [False for j in range(len(FontChecker.tagListIndex))] for i in self.fontList}
        self.fontCheckImageProducer = fontCheckImageProducer
    
    def __output__(self, ax, output):
        ax.imshow(self.fontCheckImageProducer.getFontImage(self.nowInd))
        with output:
            output.clear_output(wait=True)
            display(ax.figure)
    def __registerData__(self, checkBoxList):
        self.data[self.fontList[self.nowInd]] = [i.value for i in checkBoxList]
    
    def saveData(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.data, f)

    def showWidgets(self):
        buttonNext = widgets.Button(description='Next')
        buttonPrev = widgets.Button(description='Prev')

        checkBoxList = [ widgets.Checkbox(value= False, description = i) for i in FontChecker.tagListIndex]
        
        output = widgets.Output()
        plt.figure(figsize = (100, 30))
        ax = plt.gca()


        def onClickNext(b: widgets.Button):
            if(self.nowInd == self.fontN-1):
                return
            self.__registerData__(checkBoxList)
            self.nowInd += 1
            self.__output__(ax, output)
        
        def onClickPrev(b: widgets.Button):
            if(self.nowInd == 0):
                return
            self.__registerData__(checkBoxList)
            self.nowInd -= 1
            self.__output__(ax, output)


        buttonNext.on_click(onClickNext)
        buttonPrev.on_click(onClickPrev)
        buttonBox = widgets.Box([buttonPrev, buttonNext])
        display(buttonBox)
        columns = FontChecker.checkListColumns
        for i in range(len(FontChecker.tagListIndex)//columns):
            box = widgets.Box(checkBoxList[i*columns: (i+1)*columns])
            display(box)
        display(output)

        buttonNext.click()
