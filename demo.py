# -*- coding: utf-8 -*-
from helper import *
seed=666

def shape(M):
    if len(M)>0:
       return len(M),len(M[0])
    return 0,0
def matxRound(M, decPts=4):
    row_len,col_len=shape(M)
    for i in range(row_len):
       for j in range(col_len):
           M[i][j]=round(M[i][j],decPts)
def transpose(M):
    return [list(x) for x in zip(*M)]

def matxMultiply(A, B):
    A_row,A_col=shape(A)
    B_row,B_col=shape(B)
    print(A_row,A_col)
    print(B_row,B_col)
    if A_col == B_row:
       transB=transpose(B)
       arr=[]
       for i in A:
           row=[]
           for j in transB:
               row.append(sum([x*y for x,y in zip(i,j)]))
           arr.append(row)
       return arr
    else:
       raise ValueError

def augmentMatrix(A, b):
    if len(A)!=len(b):
        raise ValueError
    else:
        for i in range(len(A)):
            A[i]+=b[i]
        return A

def swapRows(M, r1, r2):
    M[r1],M[r2]=M[r2],M[r1]

def scaleRow(M, r, scale):
    if scale!=0:
        M[r]=[x*scale for x in M[r]]
    else:
        raise ValueError

def addScaledRow(M, r1, r2, scale):
    M[r1]=[x+y*scale for x,y in zip(M[r1],M[r2])]       
# A = generateMatrix(3,seed,singular=False)
# b = np.ones(shape=(3,1),dtype=int) # it doesn't matter
# Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
# printInMatrixFormat(Ab,padding=3,truncating=0)

# A = generateMatrix(3,seed,singular=True)
# b = np.ones(shape=(3,1),dtype=int)
# Ab = augmentMatrix(A.tolist(),b.tolist()) # 请确保你的增广矩阵已经写好了
# printInMatrixFormat(Ab,padding=3,truncating=0)

# def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
#     if len(A) != len(b):
#         raise ValueError
 
#     Ab = augmentMatrix(A,b)
 
#     for c in range(len(A[0])):
#         AbT = transpose(Ab)
#         col = AbT[c]
#         if col[c:]:
#             maxValue = max(col[c:],key=abs)
#         if abs(maxValue) < epsilon:
#             return None
 
#         maxIndex = col[c:].index(maxValue)+c
 
#         swapRows(Ab,c,maxIndex)
#         scaleRow(Ab,c,1.0/Ab[c][c])
 
#         for i in range(len(A)):
#             if Ab[i][c] != 0 and i != c:
#                 addScaledRow(Ab,i,c,-Ab[i][c])
#     matxRound(Ab)
#     return [[value] for value in transpose(Ab)[-1]]

def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A)!=len(b):
        return None
    # 生成增广矩阵
    Ab=augmentMatrix(A,b)
    # 初始化最大值
    maxValue=Ab[0][0]
    for i in range(len(A)):
        # 求出第i列对角线及以下元素绝对值的最大值，以及最大值所在的行数
        for b in range(i,len(A)):
            maxIndex=i
            currentValue=A[b][i:b+1]
            currentValue=[abs(val) for val in currentValue]
            if max(currentValue)>maxValue:
                maxIndex=b
                maxValue=currentValue
            if maxValue<epsilon:
                return None
        #初等行变换
        swapRows(Ab,i,maxIndex)
        scaleRow(Ab,i,1.0/Ab[i][i])
        for j in range(len(A)):
            if Ab[j][i] != 0 and j != i:
                addScaledRow(Ab,j,i,-Ab[j][i])
    # 保留小数位
    matxRound(Ab,decPts)
    #将得到的Ab倒置求出x
    return [[x] for x in transpose(Ab)[-1]]
# print(gj_Solve([[2,-8,3],[-4,-1,-6],[9,4,-9]],[[1],[1],[1]]))

from helper import *
from matplotlib import pyplot as plt
# %matplotlib inline

X,Y = generatePoints(seed,num=100)

# ## 可视化
# plt.xlim((-5,5))
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
# plt.show()

m2=2.0533
b2=13.4731
 
# # 不要修改这里！
# plt.xlim((-5,5))
# x_vals = plt.axes().get_xlim()
# y_vals = [m*x+b for x in x_vals]
# plt.plot(x_vals, y_vals, '-', color='r')
 
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
 
# plt.show()

# x1,x2 = -5,5
# y1,y2 = x1*m2+b2, x2*m2+b2

# plt.xlim((-5,5))
# plt.xlabel('x',fontsize=18)
# plt.ylabel('y',fontsize=18)
# plt.scatter(X,Y,c='b')
# plt.plot((x1,x2),(y1,y2),'r')
# plt.title('y = {m:.4f}x + {b:.4f}'.format(m=m2,b=b2))
# plt.show()

# def calculateMSE(X,Y,m,b):
#     if len(X) == len(Y) and len(X) != 0:
#         n = len(X)
#         square_li = [(Y[i]-m*X[i]-b)**2 for i in range(n)]
#         return sum(square_li) / float(n)
#     else:
#         raise ValueError

def calculateMSE(X,Y,m,b):
    if len(X) == len(Y):
        square_error = [(Y[i]-m*X[i]-b)**2 for i in range(len(X))]
        return sum(square_error) / len(X)
    else:
        raise ValueError
 

def linearRegression(X, Y):
    X = [[x, 1] for x in X]
    Y = [[y] for y in Y]
    XT = transpose(X)
    A = matxMultiply(XT, X)
    b = matxMultiply(XT, Y)
    h = gj_Solve(A, b)
    return h[0][0], h[1][0]

m,b = linearRegression(X,Y)
print(calculateMSE(X,Y,m2,b2))