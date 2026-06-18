# 继承自nn.module,自动带backward的效果，可以直接输出MLP的层级
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   


# result
# X = torch.rand(2,784) # 设置一个随机的输入张量
# net = MLP() # 实例化模型
# print(net) # 打印模型
# net(X) # 前向计算

# MLP(
#   (hidden): Linear(in_features=784, out_features=256, bias=True)
#   (act): ReLU()
#   (output): Linear(in_features=256, out_features=10, bias=True)
# )
# tensor([[ 0.0149, -0.2641, -0.0040,  0.0945, -0.1277, -0.0092,  0.0343,  0.0627,
#          -0.1742,  0.1866],
#         [ 0.0738, -0.1409,  0.0790,  0.0597, -0.1572,  0.0479, -0.0519,  0.0211,
#          -0.1435,  0.1958]], grad_fn=<AddmmBackward>)