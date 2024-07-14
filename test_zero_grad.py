import torch

x = torch.tensor([1., 1.], requires_grad=True)
y = 200 * x
# 定义损失
loss = y.sum()
print("x:", x)
print("y:", y)
print("loss:", loss)
print("反向传播前, 参数的梯度为: ", x.grad)
# 进行反向传播
loss.backward()
print("反向传播后, 参数的梯度为: ", x.grad)
# 定义优化器
optim = torch.optim.Adam([x], lr=0.001)  # Adam, lr = 0.001
print("更新参数前, x为: ", x)
optim.step()
print("更新参数后, x为: ", x)
# 再进行一次网络运算
y = 100 * x
# 定义损失
loss = y.sum()
# 进行optimizer.zero_grad()
optim.zero_grad()
loss.backward()  # 计算梯度grad, 更新 x*grad
print("进行optimizer.zero_grad(), 参数的梯度为: ", x.grad)