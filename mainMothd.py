import os
import numpy as np
import pre_teatment as pt
import FuzzyCMeans as cm
import re
import common as cn
import FCM as fcm

if __name__ == '__main__':
    #fuzzy C Means的聚类个数
    cate = 7
    #Fuzzy C Means 参数m
    m = 2
    #数据集分割率
    rate  = 0.6
    # 数据2维处理
    #filename = './Earthquakes/Earthquakes_TRAIN.txt'
    filename = './Strawberry/Strawberry'
    #filename = './Computers/Computers'

    f  = open('filename.txt','r')
    last_file = f.read()
    f.close()
    matchObj = re.match(last_file , filename)
    if matchObj:
        #如果数据集没变从文件提取已有数据
        pass
    else:
        #如果数据集改变写入filename后再生成新的聚类中心及隶属度数据
        ff = open('filename.txt','w')
        ff.write(filename)
        ff.close()

    Train1, Test1, Train2, Test2 = pt.dataPreTreat(filename , rate)
    TrainDataMerge = np.vstack((Train1, Train2))
    TestDataMerge = np.vstack((Test1, Test2))
    if matchObj == None:
        print('Fuzzy C Means')
        #tmpTrain = np.zeros([TrainDataMerge.shape[0], TrainDataMerge.shape[1], cate])
        # 提取概念点
        cm.FuzzyCMeans(TrainDataMerge , cate, m)
        cm.tansTestData(TestDataMerge , cate , m)

    #从文件中还原3维数组结构

    #TrainData = np.zeros([TrainDataMerge.shape[0] , TrainDataMerge.shape[1] , cate])
    #TestData = np.zeros([TestDataMerge.shape[0] , TestDataMerge.shape[1] ,cate])
    ResTrain = np.loadtxt('ResTrain.txt')
    ResTest = np.loadtxt('ResTest.txt')
    #矩阵数据转换 方便输入生成FCM
    # seq , timepoint , node
    TrainData = cn.reshapeMatrix(ResTrain , TrainDataMerge.shape[0] , TrainDataMerge.shape[1],cate)
    TestData = cn.reshapeMatrix(ResTest , TestDataMerge.shape[0] , TestDataMerge.shape[1],cate)
    TrainFcmMat = np.zeros([TrainData.shape[0], cate+1, cate])
    TestFcmMat = np.zeros([TestData.shape[0], cate+1, cate])
    #每条数据都生成一个FCM
    if not os.path.exists('TrainGCN.npy'):
        for iFCM in range(TrainData.shape[0]):
            #分解成每个node来得到FCM
            for iNode in range(cate):
                #在这里存入标签类别
                if iFCM < Train1.shape[0]:
                    TrainFcmMat[iFCM , 0]= 1
                else:
                    TrainFcmMat[iFCM, 0] = 2
                TrainFcmMat[iFCM , iNode+1]= fcm.LBFSGB(TrainData[iFCM, :, :], cate,iNode)
        print(type(TrainFcmMat))
        np.save(file = 'TrainGCN.npy',arr = TrainFcmMat)

    if not os.path.exists('TestGCN.npy'):
        for iFCM in range(TestData.shape[0]):
            #分解成每个node来得到FCM
            for iNode in range(cate):
                # 在这里存入标签类别
                if iFCM < Test1.shape[0]:
                    TestFcmMat[iFCM, 0] = 1
                else:
                    TestFcmMat[iFCM, 0] = 2
                TestFcmMat[iFCM , iNode+1] = fcm.LBFSGB(TestData[iFCM, :, :], cate,iNode)
        np.save(file = 'TestGCN.npy',arr = TestFcmMat)


    # #GCN读取保存的数据
    # TrainFcmMat = np.load(file='TrainGCN.npy')
    # TestFcmMat = np.load(file='TestGCN.npy')




