import torch

# 1. 定义自变量 x，必须设置 requires_grad=True 才能求导
x = torch.tensor(2.0, requires_grad=True)

# 2. 定义函数 y = 2x
y = 2 * x

# 3. 反向传播，自动计算梯度
y.backward()

# 4. 查看导数 dy/dx / 偏导数 
print("dy/dx =", x.grad) 