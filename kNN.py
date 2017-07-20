# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:11:06 2017

@author: liuzhao
"""

from numpy import*
import operator
from os import listdir

## 给出训练数据以及对应的类别
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels


## K-近邻算法
## inX为需要分类的目标向量；
## dataSet 为数据集
## labels 数据中对应的标签
## k 选择的最近邻的数目
def classify0(inX, dataSet, labels, k):
    dataSize = dataSet.shape[0]
    ### 计算欧式距离
    diffMat = tile(inX,(dataSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1) ##行向量分别相加，得到新行向量
    distances = sqDistances**0.5

    ##对距离排序 返回下标
    sortedDistIndicies = argsort(distances)
    
    classCount={}
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        # 对选取的K个的样本属性的类别进行统计
        classCount[voteILabel] = classCount.get(voteILabel,0) + 1
    ## 选取出现的类别次数最多的类别
    maxCount = 0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key
            
    return classes


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3)) ##创建返回的矩阵
    classLabelVector = []
    
    index = 0
    for line in arrayOfLines:
        line = line.strip()###截取所有的回车字符
        listFromLine = line.split('\t')##用tab切割
        ##选取前三个元素，存储在特征矩阵中
        returnMat[index,:] = listFromLine[0:3]
        ##将列表的最后一列存储到向量classLabelVector中
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
        
    
    
    
