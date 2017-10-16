# coding: utf-8
# 累计BP 相对于标准BP是每次把所有的数据都运行完，把所有的误差都累计起来在更新参数
import pandas as pd
import numpy as np
from pandas import *
from numpy import *

x = pd.read_csv(r'C:\Users\zmy\Desktop\titanic\xigua4.csv', header=None, nrows=8)
y = pd.read_csv(r'C:\Users\zmy\Desktop\titanic\xigua4.csv', header= None, skiprows=8,nrows=1)
y = y-1
x = x.transpose()
x = array(x)
y = array(y)
y = y.transpose()

Eta = 1 # 学习率
t = 1 # 输出
m, n = shape(x) # 数据集的行与列
w = np.random.rand(n+1, n) # 隐含层与输出层的权重
v = np.random.rand(n, n+1) # 输入层与隐含层的权重
Zta = np.random.rand(t) # 输出层阈值
Gamma = np.random.rand(n+1) # 隐含层阈值
bn = zeros((m,n+1)) # 隐含层输出
yk = zeros((m,t)) # 输出层输出
Alpha = zeros(n)

gj = zeros(m)
eh = zeros((m,n+1))

k = 0
sn = 0
old_ey = 0
while(1):
    k += 1
    ey = 0
    for i in range(0, m):
        for h1 in range(0, n+1):
            temp = 0
            for h2 in range(0, n):
                temp = temp + v[h2][h1] * x[i][h2]
            bn[i][h1] = 1 / (1+ exp(-temp+ Gamma[h1]))

        for h1 in range(0, t):
            temp = 0
            for h2 in range(0, n+1):
                temp += w[h2][h1] * bn[i][h2]
            yk[i][h1] = 1 / (1 + exp(-temp + Zta[h1]))
        # 计算累计误差
        for h1 in range(0,t):
            ey += pow(yk[i][h1] - y[i], 2) / 2

    for h1 in range(0, m):
        gj[h1] = yk[h1][0] * (1 - yk[h1][0]) * (y[h1] - yk[h1][0])
    for i in range(0, m):
        for h1 in range(n+1):
            temp = 0
            for h2 in range(0, t):
                temp += w[h1][h2] * gj[i]
            eh[i][h1] = bn[i][h1] * (1 - bn[i][h1]) * temp
    w1 = zeros((n+1, t))
    v1 = zeros((n,n+1))
    Zta1 = zeros(t)
    Gamma1 = zeros(n+1)
    # 计算四个参数的导数
    for i in range(0, m):
        for h1 in range(0, t):
            Zta1[h1] += (-1) * gj[i] * Eta
            for h2 in range(0, n+1):
                w1[h2][h1] += Eta * gj[i] * bn[i][h2]

        for h1 in range(0, n+1):
            Gamma1[h1] += Eta * (-1) * eh[i][h1]
            for h2 in range(0, n):
                v1[h2][h1] += Eta * eh[i][h1] * x[i][h2]
    # 更新参数
    v = v + v1
    w = w + w1
    Gamma = Gamma + Gamma1
    Zta = Zta + Zta1
    if (abs(old_ey-ey) < 0.0001) :
        sn += 1
        if sn == 100:
            break
    else:
        old_ey = ey
        sn = 0









for i in range(0,m):
    for j in range(0,t):
        print i,yk[i][j], y[i]