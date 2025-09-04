import torch
import torch.nn as nn
import torch.optim as optim

# 1.搭建网络
# 注意：自己编写的网络，必须继承nn.Module
class Net(nn.Module):
    def __init__(self):
        # 注意：必须调用父类的构造函数
        super(Net, self).__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=2)
        self.linear2 = nn.Linear(in_features=2, out_features=2)
        # 手动对网络参数进行初始化
        self.linear1.weight.data= torch.tensor([[0.15 ,0.20],[0.25, 0.30]])
        self.linear2.weight.data= torch.tensor([[0.40 ,0.45],[0.50, 0.55]])
        self.linear1.bias.data = torch.tensor([0.35, 0.35])
        self.linear2.bias.data = torch.tensor([0.60, 0.60])

    # 定义前向传播
    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        # 正向传播结束之层，需要返回结果
        return x
# 2.反向传播    
if __name__ == "__main__":
    # 输入数据，注意：二维列表表示批次样本的输入
    inputs = torch.tensor([[0.05, 0.10]])
    # 真实值
    targets = torch.tensor([[0.01, 0.99]])
    # 创建网络
    net = Net()
    # 这么写直接调用了对象内部的forward方法
    outputs = net(inputs)
    #print(outputs)

    # 计算误差
    loss= torch.sum((outputs - targets)**2)/2
    #print(loss)
    # 构建优化器
    optimizer = optim.SGD(net.parameters(), lr=0.5)
    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 参数更新
    optimizer.step()
    
    # 打印参数
    print(net.linear1.weight.grad.data)
    print(net.linear2.weight.grad.data)

    # 打印更新后的参数
    print(net.state_dict())
  