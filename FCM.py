from scipy.optimize import minimize
import numpy as np
import math
#输入的时间序列 、实际输出的时间序列、w为FCM的连接矩阵
def sigm(x):
    #return ((math.exp(x)-math.exp(-x)) / (math.exp(x)+math.exp(-x)))
    return 1 / (1+math.exp(-5*x))

#data_error
def fun(x ,y1, y2 , cate , inode):

    TimeData = np.zeros([y1.shape[0]])
    for i in range(y1.shape[0]):
        TimeData[i] = sigm(np.dot(y1[ i, :], x))
    err = np.subtract(TimeData,y2[ :, inode])
    sumE = np.dot(err, err)
    Obj =  sumE / y1.shape[0]
    return Obj

#设置解空间
def setBound(min,max , cate , eps):
    bound = np.zeros([cate,2])
    for i in range(cate):
        bound[i,0] = min+eps
        bound[i,1] = max-eps
    bound = tuple(bound)
    return bound


def LBFSGB(Data , cate , inode):
    y1 = Data[:-1,:]
    y2 = Data[1:,:]

    # bound = ((-1.0,1.0),  (-1.0,1.0),   (-1.0,1.0),  (-1.0,1.0),   (-1.0,1.0),    (-1.0,1.0),   (-1.0,1.0) ) # x1min, x1max, x2min, x2max
    #     # print(bound[0][0])
    #     # exit()
    eps = 1e-07
    bound = setBound(-1.0,1.0 , cate,eps)
    x0 = np.random.uniform(-1,1,size = (cate))
    #res = minimize(fun, x0, args = (y1 , y2, cate , inode) ,method='L-BFGS-B',bounds = bound , options={'disp': True, 'maxiter': 300, 'maxfun': 1500000})
    res = minimize(fun, x0, args = (y1 , y2, cate , inode) ,method='L-BFGS-B',bounds = bound , options={'disp': False, 'maxiter': 300, 'maxfun': 1500000})
    #print(res.x)
    #稀疏性处理
    result = np.copy(res.x)
    for i in range(res.x.shape[0]):
        if np.abs(res.x[i]) <= 0.05:
            result[i] = 0
        else:
            result[i] = '%0.8f' % res.x[i]
    print(result)
    return result