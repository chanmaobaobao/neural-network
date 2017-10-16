# coding: utf-8
import pandas as pd
from pandas import *
import numpy as np
from numpy import *
# 取数据，转换数据形式
# df = pd.read_csv(r"C:\Users\zmy\Desktop\titanic\watermelon.csv",skiprows=[2],header=None)
#
# df = df.transpose()
# state = ['density','rate','result']
# df.rename(columns={0:'density',1:'rate',2:'result'},inplace = True)
# # print df
# x = df[['density','rate']]
# x = array(x)
# result = df['result']
# result = array(result)

x = pd.read_csv(r"C:\Users\zmy\Desktop\titanic\xigua4.csv",header=None,nrows=8)
x = x.transpose()
x = array(x)
result = pd.read_csv(r"C:\Users\zmy\Desktop\titanic\xigua4.csv",header=None,skiprows=8,nrows=1)
result = result.transpose()
result = array(result)
result = result - 1
print x
print result

#bp算法
m,n = shape(x)
print m,n
t = 1
v = np.random.rand(n,n+1) # 输入层与隐含层的权值
w = np.random.rand(n+1, t)# 隐含层和输出层的权值
thy = np.random.rand(n+1)# 隐含层的阈值
tho = np.random.rand(t)# 输出层的阈值
out = np.zeros((m,t))# 输出层的输出值
bn = np.zeros(n+1)#隐含层的输出值
gj = np.zeros(1)
eh = np.zeros(n+1)
xk = 0.1

kn = 0 # 迭代次数
sn = 0 #
old_ey = 0

while(1):
    kn = kn + 1
    # print 'kn',kn
    ey = 0
    for i in range(0,m):
        #计算隐含层输出
        for j in range(0, n+1):
            ca = 0
            for h in range(0, n):
                ca = ca + v[h][j] * x[i][h]
            bn[j] = 1/(1+exp(-ca+thy[j]))
        # 计算输出层输出
        for h1 in range(0,t):
            ba = 0
            for h2 in range(0,n+1):
                ba = ba + w[h2][h1] * bn[h2]
            out[i][h1] = 1 / (1+ exp(-ba + tho[h1]))
        # 计算累积误差
        for h1 in range(0,t):
            ey = ey + pow((out[i][h1] - result[i]), 2)/2
            # print 'ey', ey
        # 计算gj
        for h1 in range(0,t):
            gj[h1] = out[i][h1]*(1-out[i][h1])*(result[i] - out[i][h1])
            # print out[i][h1],result[i]
        # 计算eh
        for h1 in range(0,n+1):
            for h2 in range(0, t):
                eh[h1] = eh[h1] + bn[h1] * (1 - bn[h1]) * w[h1][h2]*gj[h2]
        # 更新w
        for h2 in range(0, t):
            for h1 in range(0,n+1):
                w[h1][h2] = w[h1][h2] + xk * gj[h2] * bn[h1]
        #更新输出阈值
        for h1 in range(0,t):
            tho[h1] = tho[h1] - xk * gj[h1]
        # 更新输入层与隐含层的权值
        for h2 in range(0, n + 1):
            for h1 in range(0, n):
                v[h1][h2] = v[h1][h2] + h1 * eh[h2] * x[i][h1]
        #更新隐含层阈值
        for h1 in range(0,n+1):
            thy[h1] = thy[h1] - xk * eh[h1]
    if(abs(ey-old_ey) < 0.0001):
        # print abs(ey-old_ey)
        sn = sn + 1
        if(sn == 100):
            break

    else:
        old_ey = ey
        # ey = 0
        sn = 0


for i in range(0,m):
    for j in range(0,t):
        print i,out[i][j], result[i]







