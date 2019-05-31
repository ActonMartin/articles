---
title: 机器学习实战
date: 2019-05-31 16:28:00
tags: [编程,感悟]
categories:  编程 
---


[TOC]

*<!-- toc -->*

# 一、KNN的实现

#### 1.预测电影的种类
```
import numpy as np
import operator
def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def classify(inx, dataSet, labels, k):
    dataSetsize = dataSet.shape[0]
    diffMat = np.tile(inx, (dataSetsize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()            #argsort返回数组值的从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabels = labels[sortedDistIndicies[i]]
        #dict.get(key, default=None)
        #字典的get，返回键的值，如果不在返回None，这里是返回的数字0，
        #字典本来就是空的，当检查到没有该元素的时候，就加一，这正是为何这里的get之后需要加一
        classCount[voteIlabels] = classCount.get(voteIlabels,0)+1 
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
if __name__ == '__main__':
	group, labels = createDataSet()
	test = [1,2]
	test_class = classify(test, group, labels, 3)
	print(test_class)
```

tips:

KNN距离公式如下：
$$
L{p}(X{i},Y{i})=(\sum_{i=1}^{n}\left | X{i}^{(l)}-X{j}^{(l)} \right |^{p})^{\frac{1}{p}}
$$
对于代码值得注意的地方有   1.np.tile的使用

​								                  2.字典get的使用

