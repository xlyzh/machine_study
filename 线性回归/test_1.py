import numpy as np
import matplotlib.pyplot as plt
# 二元线性回归（西瓜书  P54）
# ---------------1. 准备数据----------
data = np.array([[32, 31], [53, 68], [61, 62], [47, 71], [59, 87], [55, 78], [52, 79], [39, 59], [48, 75], [52, 71],
                 [45, 55], [54, 82], [44, 62], [58, 75], [56, 81], [48, 60], [44, 82], [60, 97], [45, 48], [38, 56],
                 [66, 83], [65, 118], [47, 57], [41, 51], [51, 75], [59, 74], [57, 95], [63, 95], [46, 79], [50, 83]])
# 提取data中的两列数据，分别作为x，y
x = data[:, 0]
y = data[:, 1]
# 计算 w
def w_b_caculation(data):
    length=len(data)
    x_i=data[:,0]
    y_i=data[:,1]
    x_average=np.average(x_i)
    # 赋初值
    w_molecule=0
    w_denominator=0
    for i in range(0,length):
        w_molecule += y_i[i]*(x_i[i]-x_average)
        w_denominator += x_i[i]**2
    w=w_molecule/(w_denominator-length*x_average**2)
    b_sum=0
    for i in range(0, length):
        b_sum += y_i[i]-w*x_i[i]
    b=b_sum / length
    return w,b
# 计算损失值
def loss_function(w,b,data):
    x_i=data[:,0]
    y_i=data[:,1]
    length=len(data)
    loss_value = 0
    for i in range(0,length):
        loss_value += (y_i-(w*x_i[i]+b))**2
    return loss_value
# 结果展示
w,b=w_b_caculation(data)
print('线性拟合后，w=%.2f   b=%.2f'%(w,b))
loss_value=loss_function(w,b,data)
fig = plt.figure('二维拟合直线')
ax=fig.add_subplot(111)
ax.scatter(x,y)
ax.set_aspect('equal', adjustable='box')
y_after=x*w+b
ax.plot(x,y_after,'r')
plt.show()