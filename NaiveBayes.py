"""
机器学习算法-朴素贝叶斯分类器
姓名：pcb
日期：2018.12.21
"""

from numpy import *
import re
import os
import feedparser                    #从RSS源上加载数据
import operator
# -*- coding: utf-8 -*-
#-----------------------------词表到向量转换函数----------------------------------------------------

"""
创建实验样本函数
"""
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthlrss','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]                        #1代表侮辱性文字，0代表正常言论
    return postingList,classVec


"""
创建一个文档中出现的不重复列表
"""
def createVocabList(dataSet):
    vocabSet=set([])                              #创建一个空集合
    for document in dataSet:
        vocabSet=vocabSet|set(document)           #求两个集合的并集

    #将vocabSet集合按照字符串首字母的顺序排序

    #sorted(vocabSet, key=str.lower)，使用这个函数可以使得列表中的各个字符串按照首字母的顺序进行排序
    return list(vocabSet)

"""
朴素贝叶斯的词集模型，将出现的单词对应位置设置为1
"""
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)                  #创建一个和词汇表等长的向量，并将其中的元素全部设为0
    for word in inputSet:                         #遍历文档词汇
        if word in vocabList:
            returnVec[vocabList.index(word)]=1    #如果出现了词汇表中的单词，则将文档向量的对应值设置为1
        else:
            print('the word:%s is not in my Vocabulary!'% word)
    return returnVec

"""
朴素贝叶斯词袋模型
区别：可以允许单词出现多次，单词每出现一次在对应的位置上+1
"""
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec


"""
朴素贝叶斯分类训练函数
"""
def trainNB0(trainMattrix,trainCategory):
    numTrianDocs=len(trainMattrix)
    numWords=len(trainMattrix[0])
    pAbusive=sum(trainCategory)/float(numTrianDocs)

    #初始化概率，为了避免某个概率值为0，从而使得整体的概率值为0的影响
    p0Denom=2.0
    p1Denom=2.0
    p0Num=ones(numWords)
    p1Num=ones(numWords)

    #向量相加
    for i in range(numTrianDocs):
        if trainCategory[i]==1:
            p1Num+=trainMattrix[i]
            p1Denom+=sum(trainMattrix[i])
        else:
            p0Num+=trainMattrix[i]
            p0Denom+=sum(trainMattrix[i])

    p1Vect=log(p1Num/p1Denom)                     #为了防止下溢出或者浮点数舍入导致错误，而采用对数处理不会有任何损失
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)   #条件概率
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))

    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry1=['stupid','garbage']
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classifyNB(thisDoc1,p0V,p1V,pAb))

#-----------示例：使用贝叶斯分类器过滤垃圾邮件------------------------

"""
接受一个大字符串，并将其解析为字符串列表
"""
def textParse(bigString):
    regEx=re.compile('\W+')
    listOfToken=regEx.split(bigString)
    return [tok.lower() for tok in listOfToken if len(tok)>2]  #去掉少于两个字符的字符串，并将字符串改为小写

"""
对贝叶斯垃圾邮件分类器进行自动化处理
"""
def spamTest():
    docList=[];classList=[];fullText=[]

    #导入文件夹spam和ham中的文本文件
    for i in range(1,26):
        filePath1 ='email/spam/'+str(i)+'.txt'
        filePath2 = 'email/ham/' + str(i) + '.txt'
        wordList=textParse(open(filePath1).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(open(filePath2).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    #从50份邮件中应以选取10封邮件作为测试集，同时将这10封邮件从训练集中剔除
    #交叉验证
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trianMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trianMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #进行一次迭代
    p0V,p1V,pSpam=trainNB0(array(trianMat),array(trainClasses))

    #多次迭代求平均错误率
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is :',float(errorCount)/len(testSet))
    return float(errorCount)/len(testSet)

#-----------------------------------------------------------------

#----------使用朴素贝叶斯分类器从个人广告中获取区域倾向----------------

"""
计算出现频率
"""
def calcMostFreq(vocabList,fullTxet):
    fredDict={}
    #遍历词汇表的每个词，并统计它在文本中出现测次数，然后根据出现的次数从高到低对词典进行排序
    for token in vocabList:
        fredDict[token]=fullTxet.count(token)
    sortedFreq=sorted(fredDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]                                     #最后返回出现频率最高的30个单词

def localWords(feed1,feed0):
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trianMat=[];trainClasses=[]
    for docIndex in trainingSet:
        trianMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainNB0(array(trianMat),array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVextor=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVextor),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is %d',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V,float(errorCount)/len(testSet)

"""
最具表征性的词汇显示函数
"""
def getTopWords(ny,sf):
    vocabList,p0V,p1V,errorCount=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if p0V[i]>-4.5:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i]>-4.5:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])

    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

#-----------------------------------------------------------------

def main():

# #1.--------利用设定的数据集进行朴素贝叶斯分类器的测试------------------
#     testingNB()

# #2.--------示例：使用朴素贝叶斯过滤垃圾邮件---------------------------
#     #为了更好地估计错误率，将重复多次取平均值
#     averageErrorCountRate=0.0
#     for i in range(1000):
#         averageErrorCountRate += spamTest()
#     print ('the final AverageErrorCountRate is %d',float(averageErrorCountRate/1000))

# 3.--------使用朴素贝叶斯分类器从个人广告中获取区域倾向------------------
    #由于书上给的地址有问题，所以将地址改为下面的两个地址获取个人广告
    ny=feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
    sf=feedparser.parse('https://sfbay.craigslist.org/search/apa?format=rss')
    ny1=ny['entries']
    print(len(ny1))
    sf1 = sf['entries']
    print(len(sf1))

#------用于计算500次的平均误差------------------------------------------
    # averageErrorCountRate=0.0
    # for i in range(500):
    #     vocabList,pSF,pNY,errorCount=localWords(ny,sf)
    #     averageErrorCountRate+=errorCount
    # print ('the final AverageErrorCountRate is %d',float(averageErrorCountRate/500))

    #最具表征性的词汇显示函数
    getTopWords(ny,sf)

if __name__=="__main__":
    main()