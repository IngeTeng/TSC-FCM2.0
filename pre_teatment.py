import numpy as np
import matplotlib.pyplot as plt

#分割数据集
def splitData(dataset, ratio=0.6):
    len_train_data = int(len(dataset) * ratio)
    return dataset[:len_train_data , 0: dataset.shape[1] ], dataset[len_train_data: ,0:dataset.shape[1] ]

#将时间序列转化2维
def transfrom2Dimen(data):
    dataRes = np.zeros([data.shape[0], data.shape[1], 2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]-1):
            dataRes[i , j , 0] = data[i , j+1]
            dataRes[i , j , 1] = data[i , j+1] - data[ i , j]

    return dataRes


def dataPreTreat(filename, rate):
    #data = np.loadtxt('./Computers/Computers_TRAIN.txt')
    data = np.loadtxt(filename)
    #data = np.loadtxt('./BeetleFly/BeetleFly_TRAIN.txt')
    #data = np.loadtxt('./Plane/Plane_TRAIN.txt')
    #data = np.loadtxt('./Meat/Meat_TRAIN.txt')
    data1 = np.zeros([int(data.shape[0]/2) ,data.shape[1]-1])
    #data1 = np.empty(shape = [data.shape[0] ,data.shape[1]-1] , dtype=float)
    data2 = np.zeros([int(data.shape[0]/2) ,data.shape[1]-1])

    #二分类
    index1 = 0
    index2 = 0
    for i in range(data.shape[0]):

        if int(data[i,0]) == 1:
            data1[index1] = data[i,1:]
            index1 = index1 + 1

        elif int(data[i,0]) == 2:
            data2[index2] = data[i,1: ]
            index2 = index2 + 1

    #dataSet  = transfrom2Dimen(data)
    #划分训练集与测试集 3:2
    data1Train , data1Test = splitData(data1 , rate)
    data2Train , data2Test = splitData(data2 , rate)

    #分解成2维，amplitude increment

    data1Train2Dimen = transfrom2Dimen(data1Train)
    data1Test2Dimen = transfrom2Dimen(data1Test)
    data2Train2Dimen = transfrom2Dimen(data2Train)
    data2Test2Dimen = transfrom2Dimen(data2Test)

    #绘图看一下聚类
    #data1Train2Dimen中的数据
    # x1 = data1Train2Dimen[0:5 ,:,0]
    # y1 = data1Train2Dimen[0:5 ,:,1]
    # x2 = data2Train2Dimen[0:5 , :, 0]
    # y2 = data2Train2Dimen[0:5 , :, 1]
    # x = np.random.randn(1000)
    # y = np.random.randn(1000)
    #plt.scatter(x1, y1 ,s = 5 ,c='r' , marker='o')
    #plt.scatter(x2, y2 ,s = 5 ,c='b' , marker='o')
    #plt.show()
    return data1Train2Dimen , data1Test2Dimen , data2Train2Dimen , data2Test2Dimen
    # #Fuzzy C means聚类中心
    # pass