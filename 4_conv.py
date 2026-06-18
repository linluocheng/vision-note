# 互相关即卷积，同一个东西
# 实际上不用自己实现，直接nn.Conv2d就行了
# torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0)也行 


# 1
    import torch
    from torch import nn
    # 卷积运算（二维互相关）
    def corr2d(X, K): 
        h, w = K.shape
        X, K = X.float(), K.float()
        Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
        return Y

    # 二维卷积层
    class Conv2D(nn.Module):
        def __init__(self, kernel_size):
            super(Conv2D, self).__init__()
            self.weight = nn.Parameter(torch.randn(kernel_size))
            self.bias = nn.Parameter(torch.randn(1))

        def forward(self, x):
            return corr2d(x, self.weight) + self.bias


# 2
    import torch
    from torch import nn

    # 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
    def comp_conv2d(conv2d, X):
        # (1, 1)代表批量大小和通道数
        X = X.view((1, 1) + X.shape) #conv2d必须要4维度输入，(1,1,8,8)
        Y = conv2d(X)
        return Y.view(Y.shape[2:]) # 排除不关心的前两维:批量和通道


    # 注意这里是两侧分别填充1⾏或列，所以在两侧一共填充2⾏或列 (这个设计是很巧妙的，要自己先算)
    # 8+2-3+1=8,输入和输出的尺寸要一致，padding是边缘补0
    conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,padding=1)

    X = torch.rand(8, 8)
    comp_conv2d(conv2d, X).shape

    # result
    torch.Size([8, 8])

# 3 
    # 带stride的卷积层，公式是（in-k+2p）/s 下取整后 +1
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    comp_conv2d(conv2d, X).shape
    # result
    torch.Size([2, 2])
