# -*- coding: utf-8 -*-
from helper import *
seed=666
def augmentMatrix(A, b):
    if len(A)!=len(b):
        raise ValueError
    else:
        for i in range(len(A)):
            A[i]+=b[i]
        return A

A = generateMatrix(3,seed,singular=False)
b = np.ones(shape=(3,1),dtype=int) # it doesn't matter
Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
printInMatrixFormat(Ab,padding=3,truncating=0)