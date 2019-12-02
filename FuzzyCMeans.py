#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:51:45 2019
模糊c聚类:https://blog.csdn.net/lyxleft/article/details/88964494
@author: youxinlin
"""
import copy
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
global MAX  # 用于初始化隶属度矩阵U
MAX = 10000.0

global Epsilon  # 结束条件
Epsilon = 0.0000001


def print_matrix(list):
    """
    以可重复的方式打印矩阵
    """
    for i in range(0, len(list)):
        print(list[i])


def initialize_U(data, cluster_number):
    """
    这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.
    """
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U


def distance(point, center):
    """
    该函数计算2点之间的距离（作为列表）。我们指欧几里德距离。闵可夫斯基距离
    """
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    """
	结束条件。当U矩阵随着连续迭代停止变化时，触发结束
	"""
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    """
    在聚类结束时使U模糊化。每个样本的隶属度最大的为1，其余为0
    """
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def fuzzy(data, cluster_number, m):
    """
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    输入参数：簇数(cluster_number)、隶属度的因子(m)的最佳取值范围为[1.5，2.5]
    """
    # 初始化隶属度矩阵U
    U = initialize_U(data, cluster_number)
    # print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    #print("j = %d , i = %d , k = %d" %(j , i , k))
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)

        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)

        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    # 分母
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
                U[i][j] = 1 / dummy

        if end_conditon(U, U_old):
            print("已完成聚类")
            break
    #print(C)
   # U = normalise_U(U)
    return C ,U


def tansTestData(dataSet , cluster_number , m):
    data = []
    for i in range(dataSet.shape[0]):
        for j in range(dataSet.shape[1]):
            data.append(dataSet[i][j].tolist())
    U = initialize_U(data, cluster_number)
    C = np.loadtxt('CateCenter.txt')
    C = C.tolist()
    distance_matrix = []
    for i in range(0, len(data)):
        current = []
        for j in range(0, cluster_number):
            current.append(distance(data[i], C[j]))
        distance_matrix.append(current)

    for j in range(0, cluster_number):
        for i in range(0, len(data)):
            dummy = 0.0
            for k in range(0, cluster_number):
                # 分母
                dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))
            U[i][j] = 1 / dummy
    U = np.array(U)
    np.savetxt('ResTest.txt' , U , fmt='%f',delimiter=' ')
    print('save ResTest')
# if __name__ == '__main__':
#     # data = [[6.1, 2.8], [5.1, 3.4], [6.0, 1.6], [4.6,  0.2],
#     #         [ 5.7, 2.1], [ 5.8, 1.6], [6.7, 3.1, ], [6.4, 1.9],
#     #         [ 3.0,0.3], [ 6.4, 2.0], [5.2, 3.5, ], [ 5.1, 1.8],
#     #         [5.7,  1.3], [ 5.9, 2.3], [5.4,  0.4], [5.4,  0.2],
#     #         [6.6, 1.4], [5.1, 0.2], [6.0, 1.0], [7.7,  2.0],
#     #         [6.3,  1.5], [7.4,  1.9], [5.5,  0.2], [5.7,  1.2],
#     #         [5.5,  1.2], [5.2,  0.2], [4.9,  0.1], [4.6,  0.2],
#     #         [4.6,  0.2], [5.8,  1.2], [5.0,  0.2], [6.1,  1.4],
#     #         [4.7,  0.2], [6.7,  2.5], [6.5,  2.2], [5.4,  0.2],
#     #         [5.8,  1.9], [5.4, 0.4], [5.3,  0.2], [6.1,  1.8],
#     #         [7.2,  1.8], [5.5, 1.3], [5.7,  1.3], [4.9, 1.0],
#     #         [5.4,  1.5], [5.0,  0.6], [5.2,  0.1], [5.8,  0.2],
#     #         [5.4,  0.4], [6.5, 2.0], [5.5,  1.0], [5.0,  0.3],
#     #         [6.3,  1.9], [6.9 , 1.5], [6.2,  1.5], [6.3, 1.6],
#     #         [6.4,  1.5], [4.7, 0.2], [5.5,  1.1], [5.0,  1.0],
#     #         [4.4,  0.2], [4.8,  0.2], [6.3,  2.4], [5.5,  1.3],
#     #         [5.7,  2.0], [6.5,  2.0], [6.7,  1.7], [5.2,  1.4],
#     #         [6.9,  2.3], [7.2, 2.5], [4.8,  0.1], [6.3, 1.8],
#     #         [5.1,0.3], [6.9, 2.1], [5.6,  1.3], [7.7, 2.3],
#     #         [6.4,  1.3], [5.8,  1.0], [6.1,  1.4], [5.7,  1.3],
#     #         [6.2,  1.8], [4.8,  0.2], [5.6,  1.3], [6.7, 1.8],
#     #         [5.0,  0.4], [6.3,  2.5], [5.1,  0.4], [6.6,  1.3],
#     #         [5.1,  0.5], [6.3,  1.5], [6.4,  1.8], [6.2,  2.3],
#     #         [6.7,  2.4], [4.6,  0.3], [5.5,  0.2], [5.6, 1.3],
#     #         [5.6,  2.0], [6.2, 1.3], [7.0,  1.4], [5.0,  0.2],
#     #         [4.3,  0.1], [7.7,  2.2], [5.6,  1.5], [5.8, 1.9],
#     #         [5.8,  2.4], [4.9,  0.1], [5.7,  0.3], [7.1,  2.1],
#     #         [5.1,  0.4], [6.3,  1.8], [6.7,  2.3], [5.1,  1.1],
#     #         [7.6,  2.1], [4.5,  0.3], [4.9,  0.2], [6.5,  1.5],
#     #         [5.7,  0.4], [6.8,  2.1], [4.9,  1.7], [5.1,  0.3],
#     #         [6.5,  1.8], [5.7, 1.0], [5.1,  0.2], [5.9,  1.5],
#     #         [6.4,  2.3], [4.4, 0.2], [6.1,  1.3], [6.3,  1.3],
#     #         [5.0,  1.0], [5.0,  0.2], [5.9,  1.8], [6.4,  2.2],
#     #         [6.1,  1.4], [5.6,  1.1], [6.0,  1.6], [6.0, 1.8],
#     #         [6.4,  2.1], [6.0,  1.5], [5.8,  1.2], [7.7,  2.3],
#     #         [5.0,  0.2], [6.9,  2.3], [6.8,1.4], [4.8,  0.2],
#     #         [6.7,  1.5], [4.9,  0.1], [7.3, 1.8], [4.4,  0.2],
#     #         [6.0,  1.5], [5.0,  0.2]]
#     start = time.time()
#
#     # 调用模糊C均值函数
#     res_C , res_U = fuzzy(data, 4, 2)
#     res_C = np.array(res_C)
#     data = np.array(data)
#     plt.scatter(data[:,0],data[:,1], s=5, c='r', marker='o')
#     for i in range(res_C.shape[0]):
#         plt.scatter(res_C[i,0],res_C[i,1], c='b', marker='o')
#     plt.show()
#     # 计算准确率
#     print("用时：{0}".format(time.time() - start))


def FuzzyCMeans(dataSet , number , m):
    start = time.time()
    #data是二维的
    data = []
    for i in range(dataSet.shape[0]):
        for j in  range(dataSet.shape[1]):
            data.append(dataSet[i][j].tolist())
    # 调用模糊C均值函数
    res_C, res_U = fuzzy(data, number, m)
    res_C = np.array(res_C)
    res_U = np.array(res_U)
    np.savetxt('ResTrain.txt' , res_U , fmt='%f',delimiter=' ')
    np.savetxt('CateCenter.txt' , res_C , fmt='%f',delimiter=' ')
    print('save Center and ResTrain')
    #data = np.array(data)
    #恢复输入数据的结构

    #绘制图表

    # plt.scatter(data[:, 0], data[:, 1], s=5, c='r', marker='o')
    # for i in range(res_C.shape[0]):
    #     plt.scatter(res_C[i, 0], res_C[i, 1], c='b', marker='o')
    # plt.show()
