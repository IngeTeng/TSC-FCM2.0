import numpy as np
# 源二维矩阵  ， seq ,len
def reshapeMatrix(Res , SeqNum , LenNum , cate):
    Data = np.zeros([SeqNum, LenNum, cate])
    start = 0
    end = 0 + LenNum
    for i in range(SeqNum):
        Data[i] = Res[start:end]
        start = start + LenNum
        end = start + LenNum
    return Data