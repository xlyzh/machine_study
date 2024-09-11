import numpy as np
import matplotlib.pyplot as plt
# 简单梯度下降算法
# --------------1. 定义损失函数(均方误差)--------------
def loss_function(data,w,b):
    x=data[:,0]
    y=data[:,1]
    M=len(data)
    total_cost = 0
    for i in range(0,M):
        total_cost += (y[i]-x[i]*w-b)**2
    return total_cost/M
# --------------2. 定义核心梯度下降算法函数-----
def grad_desc(data,initial_w,initial_b,alpha,num_iter):
    x=data[:,0]
    y=data[:,1]
    loss_value=[]
    w_previous, b_previous=initial_w,initial_b
    count_num=0
    for i in range(0,num_iter):
        w_current,b_current=step_grad_desc(w_previous,b_previous,alpha,data)
        loss_value.append(loss_function(data, w_current, b_current))
        count_num += 1
        # 判断是否跑''过''最优点
        if abs(loss_function(data,w_current,b_current)-loss_function(data,w_previous,b_previous))>=3:
            w_previous, b_previous = w_current,b_current
        else:
            break
    return w_current,b_current,loss_value,count_num
# 迭代更新函数
def step_grad_desc(current_w, current_b, alpha, data):
    M=len(data)
    s_w=0
    s_b=0
    for i in range(0,M):
        s_w += (y[i]-current_w*x[i]-current_b)*x[i]
        s_b += (y[i]-current_w*x[i]-current_b)
    diff_w=-2*s_w/M  # 斜率
    diff_b=-2*s_b/M  # 斜率
    update_w=current_w-diff_w*alpha
    update_b=current_b-diff_b*alpha
    return update_w,update_b
# --------------3. 定义模型的超参数------------
alpha = 0.0001  # 学习率（步长）
initial_w = 0  # 斜率初始值
initial_b = 0  # 截距初始值
num_iter = 20  # 迭代次数
# ------------4. 测试：运行梯度下降算法计算最优的w和b-------
data = np.array([[32, 31], [53, 68], [61, 62], [47, 71], [59, 87], [55, 78], [52, 79], [39, 59], [48, 75], [52, 71],
                 [45, 55], [54, 82], [44, 62], [58, 75], [56, 81], [48, 60], [44, 82], [60, 97], [45, 48], [38, 56],
                 [66, 83], [65, 118], [47, 57], [41, 51], [51, 75], [59, 74], [57, 95], [63, 95], [46, 79],
                 [50, 83]])
x = data[:, 0]
y = data[:, 1]
w,b,loss_value,count_num = grad_desc(data,initial_w,initial_b,alpha,num_iter)
print(f'w={w:.4f} \n'
      f'b={b:.4f} \n'
      f'loss_value={loss_value} \n'
      f'count_num={count_num:.0f}')
# ------------5.出图-------------------------
# 拟合曲线图
fig_1=plt.figure('拟合曲线')
ax_1=fig_1.add_subplot()
ax_1.scatter(x,y)
y_match=x*w+b
ax_1.plot(x,y_match,'r')
# 损失值变化趋势
fig_2=plt.figure('损失值变化趋势')
ax_2=fig_2.add_subplot()
x_count=np.linspace(1,10,count_num)
ax_2.scatter(x_count,loss_value)
plt.show()