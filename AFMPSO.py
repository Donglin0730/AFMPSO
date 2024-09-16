#库的导入
import numpy as np
import random
import matplotlib.pyplot as plt
from CEC2022 import cec2022_func
import pandas as pd
import time

eps = 1e6
fx_n = 1 #公式
N = 500   #种群数量
iterators = 400    #迭代次数
dim=10
w=0.9   #惯性因子
rangepop=[-100,100]    #取值范围
parts = 5 
CEC = cec2022_func(func_num = fx_n)

#待求解问题
def function(x):
    y = CEC.values(x)
    return y.ObjFunc

#两个加速系数
# c1=np.full(iterators,2.5)
# c2=np.full(iterators,0.5)
# Cmax = 2.5
# Cmin = 0.5
# r = np.full(iterators,0.5)
# r[0] = np.random.randn()
c1=np.full(N,2.0)
c2=np.full(N,2.0)
fitness=np.zeros(N)

#粒子初始化
x = np.random.uniform(low=rangepop[0], high=rangepop[1], size=(N, dim))
print(x)
v = np.random.uniform(-1, 1, size=(N, dim))
v2 = np.zeros((N, dim))
fitness = function(np.transpose(x[0:N, :])) #矩阵 x 的前 N 行进行转置

Fhistory = np.full((N, 25), eps)
FZhistory = np.full((N, 24), eps)
Xhistory = np.zeros((N , 25, dim))
XZhistory = np.zeros((N , 24, dim))

#Xgbest,Fgbest分别表示种群历史最优个体和适应度值
Xgbest,Fgbest=x[fitness.argmin()].copy(),fitness.min()
#poppn,bestpn分别存储个体历史最优位置和适应度值
Xpbest,Fpbest=x.copy(),fitness.copy()
#poppn,bestpn分别存储个体历史最优位置和适应度值
Xpbesthis,Fpbesthis=x.copy(),fitness.copy()
#bestfitness用于存储每次迭代时的种群历史最优适应度值
bestfitness=np.zeros(iterators)
l = np.zeros(N)

def GetCentroid(points):
    return np.mean(points, axis=0)

datas = []

start_time = time.time()
#开始迭代
for t in range(iterators):
    print("generation:",t)
    w= 2/(t+1)
    # r[t+1] = np.random.rand(0.9, 1.08)*f(r[t])
    # c1[t] = Cmax - r[t]*t/iterators
    # c2[t] = Cmin + r[t]*t/iterators
    StudyZu = []
    rankings = np.argsort(fitness)
    for i in range(int(N/(t+1))):
        StudyZu.append(x[rankings[i]].copy())
    StudyedModel = GetCentroid(np.array(StudyZu))
    for i in range(N):
        r1 = np.random.rand()
        r2 = np.random.rand()
        r3 = np.random.rand()
        # if l[i] < 1e-6:
        #     v[i]=w*v[i]+c1[i]*r1*(Xpbesthis[i]-x[i])+c2[i]*r2*(Xgbest-x[i])+r3*(StudyedModel-x[i]) + w/2*v2[i]
        # else:
        v[i]=w*v[i]+c1[i]*r1*(Xpbesthis[i]-x[i])+c2[i]*r2*(Xgbest-x[i])+r3*(StudyedModel-x[i]) 
        #计算新的位置
        x[i]=x[i]+v[i]
        #确保更新后的位置在取值范围内
        x[x<rangepop[0]]=np.random.uniform(-100, 100) #对矩阵的每一行进项修正
        x[x>rangepop[1]]=np.random.uniform(-100, 100)
        # print(x)
    
    #计算适应度值
    fitness = function(np.transpose(x[0:N, :]))
    # print(fitness)
    for i in range(N):
        #更新个体历史最优适应度值
        if fitness[i]<Fpbest[i]:
            Fpbest[i]=fitness[i]
            Xpbest[i]=x[i].copy()
        if fitness[i]<Fhistory[i].min():
            Fpbesthis[i]=fitness[i]
            Xpbesthis[i]=x[i].copy()
        else:
            Xpbesthis[i]=Xhistory[i][Fhistory[i].argmin()].copy()

        if np.any(Fhistory[i] == eps) == False:
            FZhistory[i] = np.delete(Fhistory[i], 0)
            Fhistory[i] = np.append(FZhistory[i], fitness[i])
            XZhistory[i] = np.delete(Xhistory[i], 0, 0)
            Xhistory[i] = (np.append(XZhistory[i], x[i])).reshape(Xhistory[i].shape)
        else:
            j = np.where(Fhistory[i] == eps)[0][0]
            Fhistory[i][j] = fitness[i]
            Xhistory[i][j] = x[i]

    #更新种群历史最优适应度值
    if Fpbest.min()<Fgbest:
        Fgbest=Fpbest.min()
        Xgbest=Xpbest[Fpbest.argmin()].copy()
    
    # for i in range(N):
    #     l[i] = np.linalg.norm(x[i]-Xhistory[i][-1])
    #     if l[i] < 1e-6:
    #         v2[i] = np.random.randn(1, 10)
    #         norm = np.linalg.norm(v2)
    #         v2[i] = v2[i] / norm * l[i]
    bestfitness[t]=Fgbest
    print("the best fitness is:",bestfitness[t])

end_time = time.time()

execution_time = end_time - start_time
print("代码块执行时间: " + str(execution_time) + " 秒")

df = pd.DataFrame(bestfitness)
df.to_excel('outputT20.xlsx', index=False)
