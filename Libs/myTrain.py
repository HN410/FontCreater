from .mypSp import *
from .myFontLib import *
from .myFontData import *
from .myLoss import *
import random
import time
import gc
import shutil

GENERATOR_NAME = "style_gen"
ENCODER_CONV_NAME = "encode_convs"
ENCODER_MAP_NAME =  "map2styles"
INTERMIDIATE_IMAGE_N = 10
G_LOSS_TYPE = ["C", "M", "R", "S"]

# ドロップアウト率の下限，．
d_dropout_limit = 0.75

# 出力を見たい中間層とその名前を得る
def getGenIntermidiateLayers(model: MyPSP):
    ans = []
    for name, innerModel in zip(["CharaE", "StyleE"], [model.chara_encoder, model.style_encoder]):
        innerModel = innerModel._blocks
        for i in range(0, len(innerModel)-1, 2):
            ans.append([name + "/Blocks/" + str(i), innerModel[i]])
    for i in range(len(model.style_encoder.map2styles[0].convs)-1):
        ans.append(["Map2Styles/" + str(i), model.style_encoder.map2styles[0].convs[i]])
    innerModel = model.style_gen.synthesis_module.blocks
    for i in range(len(innerModel)-1):
        ans.append(["Generator/" + str(i), innerModel[i]])
    return ans

# Discriminatorの中間層を出力
def getDisIntermidiatelayers(model: Discriminator3):
    ans = []
    innerModel = model.discriminator._blocks
    for i in range(0, len(innerModel)-2, 2):
        ans.append(["Discriminator/" + str(i), innerModel[i]])
    return ans

# モデルのパラメータのうち，更新対象となるものを得る
def getUpdatedParams(myPSP, forCharaEncode):

    paramUpdatedGen = []
    paramUpdatedEncMain = []
    paramUpdatedEncConvMap = []
    for name, param in myPSP.named_parameters():
        if(not forCharaEncode and (name.startswith("chara_encoder"))): 
            param.requires_grad = forCharaEncode
        else:
            param.requires_grad = True
            if GENERATOR_NAME in  name:
                paramUpdatedGen.append(param)
            else:
                if(ENCODER_CONV_NAME in name or ENCODER_MAP_NAME in name):
                    paramUpdatedEncConvMap.append(param)
                else:
                    paramUpdatedEncMain.append(param)
    return [paramUpdatedGen, paramUpdatedEncMain, paramUpdatedEncConvMap]

# 現在のドロップアウト率，正解率をもとに次のepochのdropout率を定める
def getNextDropout(dropout, correctRate):
    if(correctRate >= 0.55 and correctRate <= 0.65):
        return dropout
    elif(correctRate >= 0.65 and correctRate <= 0.85):
        return min(dropout + 0.01 , 0.9)
    elif(correctRate < 0.55):
        return max(dropout - 0.01 , 0.0)
    else:
        return min(dropout + 0.02 , 0.9)

def getNextDropoutSafe(dropout, correctRate, count):
    count = count + 1
    if(correctRate >= 0.65 and count >= 3):
        return 0, min(dropout + 0.005 , 0.99)
    if(correctRate >= 0.6):
        return 0, dropout
    elif(correctRate >= 0.9):
        return 0, min(dropout + 0.1, 0.99)
    elif(correctRate > 0.8):
        return 0, min(dropout + 0.02, 0.99)
    elif(correctRate > 0.7 and count >= 2):
        return 0, min(dropout + 0.01, 0.99)
    # elif(correctRate < 0.55 and count >= 5 and dropout > 0.05):
    #     return dropout / 2
    elif(correctRate < 0.55 and count >= 10):
        return 0, max(dropout - 0.005, d_dropout_limit)
    elif(correctRate >= 0.55 and count >= 7):
        return 6, dropout
    else:
        return count,  dropout 

# tensorboardで確認する為にGeneratorの微分値を出力
def writeGeneratorGradients(model, writer, epoch, styleDis = None):
    models = [model.chara_encoder, model.style_encoder, model.style_gen, styleDis]
    modelName = ["Chara/", "StyleEnc/", "StyleGen/", "StyleDis/"]
    for modelE, mName in zip(models, modelName):
        if(modelE is None):
            continue
        for name, param in modelE.named_parameters():
            if(param.grad is not None):
                writer.add_histogram("Gen/" + mName + name, param.grad, epoch)

# tensorboardで確認する為にDisciminatorの微分値を出力
def writeDiscriminatorGradients(model, writer, epoch):
    for name, param in model.named_parameters():
            if(param.grad is not None):
                writer.add_histogram("Dis/" + name, param.grad, epoch)
    # for e in discriminatorWeightName:
    #     writer.add_histogram("Dis/" + e, model.state_dict()[e].grad, epoch)

# checkpointからデータをロードする
def loadCheckpoints(checkpointFile, models, optimizers, dCheck, charaDisCheckpointFile,\
    inheritOnlyModel, forUnderTraining, trainCharaDis):
    myPSP, D, styleDis, charaDis = models
    optimizer, optimizer_d, optimizer_styleDis, optimizer_charaDis = optimizers

    start=0
    if os.path.exists(checkpointFile):
        checkpoint = torch.load(checkpointFile)
        if not inheritOnlyModel:
            start = checkpoint["epoch"]+1
            optimizer.load_state_dict(checkpoint["optStateDict"])
            optimizer_d.load_state_dict(checkpoint["optDStateDict"])
        myPSPDict = checkpoint["modelStateDict"]
        myPSP.load_state_dict(myPSPDict, strict=False)
        optimizer_styleDis.load_state_dict(checkpoint["optSDStateDict"])
        styleDis.load_state_dict(checkpoint["styleDiscriminatorStateDict"])
        if(not forUnderTraining):
            
            if(dCheck != ""):
                checkD = torch.load(dCheck)
                D.load_state_dict(checkD["discriminatorStateDict"], strict = False)
                optimizer_d.load_state_dict(checkD["optDStateDict"])
            else:
                if(trainCharaDis):
                    optimizer_charaDis.load_state_dict(checkpoint["optCDStateDict"])
                D.load_state_dict(checkpoint["discriminatorStateDict"], strict = False)
            if(charaDisCheckpointFile):
                checkD = torch.load(charaDisCheckpointFile)
                charaDis.load_state_dict(checkD["charaDiscriminatorStateDict"])
            else:    
                charaDis.load_state_dict(checkpoint["charaDiscriminatorStateDict"])
    return start

# optimizerを得る
def getOptimizers(models, forCharaTraining, trainCharaDis, d_optimFun, optimizer_d_lr):
    myPSP, D, styleDis, charaDis = models
    params = getUpdatedParams(myPSP, forCharaTraining)
    optimizer = torch.optim.Adam([
        {"params": params[0], "lr": 1e-4},
        {"params": params[1], "lr": 1e-5}, 
        {"params": params[2], "lr": 1e-5}
    ], lr = 1e-4, betas=(0.0, 0.99), eps = 1e-8)

    optimizer_d = None
    optimizer_d = d_optimFun(D.parameters(), optimizer_d_lr, [0.0, 0.99])
    loss_criterion = nn.BCEWithLogitsLoss(reduction = "mean")

    optimizer_charaDis = None
    if(trainCharaDis):
        optimizer_charaDis = torch.optim.Adam(charaDis.parameters(), 1e-3, [0.0, 0.99])
    optimizer_styleDis = torch.optim.Adam(styleDis.parameters(), 1e-5, [0.0, 0.99])

    return optimizer, optimizer_d, optimizer_styleDis, optimizer_charaDis 


FAKES_BACK_LOG_PATH = "cpts/fakes_back_log"
FAKES_BACK_LOG_PATH_BODY = "cpts/backlog/fakes_back_log{}"
FAKES_BACK_LOG_N = 40
FAKES_BACK_LOG_KEY = "data"

# BackLogを使っての訓練
def trainWithBackLog(phase, device, D, optimizer_d, noiseP, useWSGradient ):
    discriminator_problems_n_b = 0
    discriminator_correct_n_b = 0 
    print("BackLog")
    # どれからロードするか決定
    nowFakesBackInd = random.randint(0, FAKES_BACK_LOG_N-1)
    nowFakesBackPath = FAKES_BACK_LOG_PATH_BODY.format(nowFakesBackInd)
    fakesBackLog = torch.load(nowFakesBackPath)[FAKES_BACK_LOG_KEY]
    initFakesBackLogLen = len(fakesBackLog)

    for i_ in range(1):
        epochDLoss = 0
        iteration = 0
        for i in range(initFakesBackLogLen):
            if(random.random() < 0.5):
                continue
            data = fakesBackLog[i]
            minibatch_size = data[0].size()[0]
            # label_fake = (torch.zeros((minibatch_size, )) + 0.3 * torch.rand((minibatch_size, )) ).to(device)
            beforeCharacter = data[0].to(device, non_blocking=True)
            afterCharacter = data[1].to(device, non_blocking=True)
            fakes = data[2]
            teachers = data[3] # ver=4から差分をとらない
            fakes = fakes.to(device, non_blocking=True)
            teachers = teachers.to(device, non_blocking=True)
            beforeCharacter, afterCharacter, fakes, teachers = MyPSPAugmentation.getNoisedImages([beforeCharacter, afterCharacter, fakes, teachers], noiseP, device)
            alpha = torch.ones((1, 1))
            alpha = alpha.to(device, torch.float32, non_blocking=True)

            if(minibatch_size > 2):
                # ここでメモリをよく使うため，minibatchを小さくしておく
                if(useWSGradient):
                    if(minibatch_size > 2):
                        minibatch_size = 2
                    else:
                        minibatch_size = 1
                    if(teachers.shape[1] > 3):
                        teachers = teachers[:, : 2]
                else:
                    minibatch_size = 4
                # beforeCharacterN = beforeCharacterN[:minibatch_size]
                fakes = fakes[:minibatch_size]
                afterCharacter = afterCharacter[:minibatch_size]
                teachers = teachers[:minibatch_size]
                
                with torch.set_grad_enabled(True):

                    d_loss_back, discCorrectN_b, lossList_b, tcorrect_b, fcorrect_b = d_wgan_loss2(D, None, afterCharacter,\
                    fakes, teachers, alpha, phase, useGradient=useWSGradient, useBefore=False)
                    
                    # Discriminator loss
                    epochDLoss += d_loss_back.item()
                    d_loss_back.backward()
                    optimizer_d.step()
                    optimizer_d.zero_grad()
                
                    discriminator_problems_n_b += minibatch_size*2
                    discriminator_correct_n_b +=  discCorrectN_b
                    
                    del beforeCharacter, afterCharacter, teachers, fakes, data, minibatch_size, alpha, d_loss_back, discCorrectN_b, lossList_b, tcorrect_b, fcorrect_b
                
                # gc.collect()
                # torch.cuda.empty_cache()
            print("\riter {:4}/{}".format(iteration ,len(fakesBackLog)-1), end="")
            iteration += 1
    if(initFakesBackLogLen > 0 and discriminator_problems_n_b > 0):
        print("BackLog correct rate = {:4}".format(discriminator_correct_n_b / discriminator_problems_n_b))
    return epochDLoss, fakesBackLog, nowFakesBackPath, [discriminator_problems_n_b, discriminator_correct_n_b]

# 中間層出力用の関数を得る
def getIntermidiateHandlers(genIntermidiateList, disIntermidiateList, writer, iteration, epoch, \
    lookIntermidiate, checkGradNow,  forUnderTraining):
    Dhandles = []
    Ghandles = []
    handlePair = [Ghandles]
    intermidiateLists = [genIntermidiateList]
    if(not forUnderTraining):
        handlePair.append(Dhandles)
        intermidiateLists.append(disIntermidiateList)
    if(lookIntermidiate and iteration == 0 and (epoch% 10 == 0 or checkGradNow)):
        for handles, intermidiateList in zip(handlePair, intermidiateLists):
            for name, layer in intermidiateList:
                # forward_hookを関数を通して得る(こうしないとnameが変わらない)
                def getForwardHook(name):
                    def forward_hook(module, input, outputs):
                        # global writer
                        if(outputs.size()[1] > INTERMIDIATE_IMAGE_N):
                            writer.add_images(name, outputs[0][:INTERMIDIATE_IMAGE_N].unsqueeze(-1), epoch, dataformats="NHWC")
                        else:
                            writer.add_images(name, outputs[0].unsqueeze(-1), epoch, dataformats="NHWC")
                    return forward_hook
                handle = layer.register_forward_hook(getForwardHook(name))
                handles.append(handle)
    return Dhandles, Ghandles

# Generatorに順伝播させる関数
def forwardG(myPSP, styleDis, charaDis, charaDisLoss, beforeCharacter, teachers, afterCharacter,\
            alpha, styleLabel, GLossDict, factors, \
            forCharaTraining, forStyleTraining):
    SquareLossFactor, fakeRawFactor, styleLossFactor, charaDisFactor = factors
    featureT = fakes = None
    iterGLoss = 0
    if(forCharaTraining):
        featureT, fakeRaw, fakes = myPSP(beforeCharacter, None, alpha)
        iterGLoss = SquareLossFactor* MyPSPLoss(onSharp=0, rareP=4, separateN=8, hingeLoss=0)(fakes, beforeCharacter)
        iterMLoss = iterGLoss.item()
        GLossDict["M"] +=  iterMLoss
        fakeRaw = fakeRaw ** 2
        fakeRaw = ((fakeRaw > 3) * fakeRaw)
        iterGLoss +=  fakeRawFactor * fakeRaw.sum()
        featureT_ = featureT ** 2
        featureT_ = ((featureT_ > 3) * featureT_)
        iterGLoss += 0.2 *fakeRawFactor * featureT_.sum()
        GLossDict["R"] += iterGLoss.item() - iterMLoss
        featureT = featureT.detach()
    elif(forStyleTraining):
        featureT, style, fakeRaw,  fakes = myPSP(beforeCharacter, teachers, alpha)
        del featureT, fakeRaw
        styleOut, rawStyleOut = styleDis(style)
        styleOut = styleLossFactor * (myCrossE(styleOut,styleLabel) + 0.001 * (((((rawStyleOut > 2.0) + (rawStyleOut < -2.0)) * rawStyleOut) ** 2).mean()))
        iterGLoss = iterGLoss + styleOut
        GLossDict["S"] += iterGLoss.item()
        iterGLoss += 1 * (style ** 2).mean()
        GLossDict["R"] += iterGLoss.item() - styleOut.item()
        del style
    else:
        featureT, style, fakeRaw,  fakes = myPSP(beforeCharacter, teachers, alpha)
        featureT = featureT.detach()
        iterGLoss = SquareLossFactor * torch.nn.MSELoss()(fakes.mean([1, 2, 3]), afterCharacter.mean([1, 2, 3]))
        iterMLoss = iterGLoss.item()
        GLossDict["M"] += iterMLoss
        featureO = charaDis(fakes)
        iterGLoss  = iterGLoss + charaDisFactor *  charaDisLoss(featureO, featureT)
        iterMCLoss = iterGLoss.item()
        GLossDict["C"] += iterMCLoss - iterMLoss
        styleOut, rawStyleOut = styleDis(style)
        styleOut = styleLossFactor * (myCrossE(styleOut,styleLabel) + 0.001 * (((((rawStyleOut > 2.0) + (rawStyleOut < -2.0)) * rawStyleOut) ** 2).mean()))
        iterGLoss = iterGLoss + styleOut
        iterGLoss += 1 * (style ** 2).mean()
        iterMCSLoss = iterGLoss.item()
        GLossDict["S"] += iterMCSLoss - iterMCLoss
        fakeRaw = fakeRaw ** 2
        iterGLoss += fakeRawFactor *((fakeRaw > 2.0) * fakeRaw).sum() / 30
        iterGLoss +=  0.1 * fakeRawFactor * fakeRaw.mean()
        GLossDict["R"] +=  iterGLoss.item() - iterMCSLoss
        del featureO, fakeRaw, style, styleLabel, styleOut
    return iterGLoss, featureT, fakes

def printResults(d_loss, discriminator_ns, g_loss, GLossDict, c_loss, epochDLossList, TCorrectN,  epochStartTime, \
    epoch, forUnderTraining, trainD
    ):
    d_correct_rate = 0
    discriminator_correct_n , discriminator_problems_n = discriminator_ns[0]
    discriminator_correct_n_b , discriminator_problems_n_b = discriminator_ns[1]
    if(not forUnderTraining and trainD):
        # epochごとの正解率
        d_correct_rate = discriminator_correct_n / discriminator_problems_n
        print("EpochGLossC ... {:4f}, EpochGLossM ... {:4f}, EpochGLossS ... {:4f}, EpochGLossR ... {:4f}".format(\
            GLossDict["C"], GLossDict["M"], GLossDict["S"], GLossDict["R"]))
        d_correct_rate = discriminator_correct_n / discriminator_problems_n
        print('Epoch_D_Correct (only Now): {:.4f},  TCorrect: {:4f} '.format(d_correct_rate,\
                                        TCorrectN / discriminator_problems_n  ))
        print("DLossList: {:4f}, {:4f}, {:4f}".format(epochDLossList[0], epochDLossList[1], epochDLossList[2]))
        discriminator_correct_n += discriminator_correct_n_b
        discriminator_problems_n += discriminator_problems_n_b
        d_correct_rate = discriminator_correct_n / discriminator_problems_n
    else:
        print("epochRLoss {:4f}".format(g_loss - GLossDict["M"]))
        # print("trainGDAcc {:4f}".format(epochtrainGDAcc / epochtrainGn))
        print("EpochGLossC ... {:4f}, EpochGLossM ... {:4f}, EpochGLossS ... {:4f}, EpochGLossR ... {:4f}".format(\
            GLossDict["C"], GLossDict["M"], GLossDict["S"], GLossDict["R"]))
    epochFinishTime = time.time()
    print('-----')
    print('epoch {} || Epoch_GLoss:{:.4f}, Epoch_DLoss:{:.4f},Epoch_D_Correct: {:.4f} , Epoch_CDLoss{:.4f}'.format(
        epoch, g_loss, d_loss, d_correct_rate, c_loss))
    print('timer:  {:.4f} sec.'.format(epochFinishTime - epochStartTime))
    return d_correct_rate

# writerに結果をoutput
def outputWriter(writer,d_loss, d_correct_rate, g_loss, GLossDict, c_loss, phase, epoch, \
        forUnderTraining, trainD):
    if(phase == "train"):
        writer.add_scalar("lossG/train", g_loss, global_step=epoch)
        for key in G_LOSS_TYPE:
            writer.add_scalar("lossGList/" + key, GLossDict[key], global_step=epoch)
        writer.add_scalar("charaD/train", c_loss, global_step=epoch)
        if(not forUnderTraining):
            if(trainD):
                writer.add_scalar("lossD/train", d_loss, global_step=epoch)
                writer.add_scalar("correctD/train", d_correct_rate, global_step=epoch)
    else:
        # logs[1].append(loss)
        writer.add_scalar("lossG/valid", g_loss, global_step=epoch)
        writer.add_scalar("charaD/valid", c_loss, global_step=epoch)
        if(not forUnderTraining):
            writer.add_scalar("lossD/valid", d_loss, global_step=epoch)
            writer.add_scalar("correctD/valid", d_correct_rate, global_step=epoch)

# dropout率をupdateする
def updateDropout(D, optimizer_d, disIntermidiateList, writer, checkpoint, checkpointFile, train_d_correct,
         nowDropout, dropoutChangeCount, epoch,\
        trainRate,  trainRateC, device, d_optimFun, optimizer_d_lr,\
        trainD, emergencySave, forUnderTraining, changeDropout): 

    dropoutChangeCount, nextDropout = getNextDropoutSafe(nowDropout, train_d_correct, dropoutChangeCount)
    writer.add_scalar("D_dropout", nextDropout, global_step=epoch)
    torch.cuda.empty_cache()
    gc.collect()
    if(train_d_correct > 0.65 and not emergencySave):
        emergencySave = True
        torch.save(checkpoint, "cpts/emergency.cpt")
        shutil.copy("cpts/before.cpt", "cpts/emergency_before.cpt")
    else:
        torch.save(checkpoint, checkpointFile)
    if(trainD):
        dropoutChangeCount, nextDropout = getNextDropoutSafe(nowDropout, train_d_correct, dropoutChangeCount)
        writer.add_scalar("D_dropout", nextDropout, global_step=epoch)
        if(nextDropout > nowDropout):
            trainRate = min(trainRate + 1, 7)
            trainRateC = (epoch) % trainRate
        if(nextDropout != nowDropout and (not forUnderTraining) and changeDropout):
            D = Discriminator4(dropout_p=nextDropout).to(device)
            nowDropout = nextDropout
            checkpoint = torch.load(checkpointFile)
            D.load_state_dict(checkpoint["discriminatorStateDict"], strict = False)
            D.train()
            optimizer_d = d_optimFun(D.parameters(), optimizer_d_lr, [0.0, 0.99])
            optimizer_d.load_state_dict(checkpoint["optDStateDict"])
            disIntermidiateList = getDisIntermidiatelayers(D)
            dropoutChangeCount = 0
    torch.cuda.empty_cache()
    gc.collect()
    return D, optimizer_d,  disIntermidiateList, dropoutChangeCount, nowDropout, trainRate, trainRateC, emergencySave


# Generatorの損失をまとめるディクショナリを作成
def initGLossDict():
    return {key: 0 for key in G_LOSS_TYPE}