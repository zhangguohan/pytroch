import torch
import torch.nn as nn
import torch.optim as optim


# 1.创建以及使用dropout
def test01():
    # 创建dropout对象
    dropout = nn.Dropout(p=0.8)
    # 创建数据
    input = torch.randint(1,10,size=(5,8)).float()
    print(input)
    # 使用dropout
    output = dropout(input)
    print(output)

# 2.使用dropout 随机丢弃对网络参数的影响
def test02():
    #
    # 固定随机数种子
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn(15,1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0,10,size=(5,15)).float()

    # 计算梯度
    y = x @ w
    y.sum().backward()
    print('档度：', w.grad.reshape(1,-1).square().numpy())
   

 #
def test03():
    torch.manual_seed(0)
    # 初始化权重
    w = torch.randn(15,1, requires_grad=True)
    # 初始化输入数据
    x = torch.randint(0,10,size=(5,15)).float()
    # 初始化丢弃层
    droput = nn.Dropout(p=0.8)
    x = droput(x)
    y = x @ w
    y.sum().backward()
    print('梯度：', w.grad.reshape(1,-1).square().numpy())




if __name__ == '__main__':
    test02()
    print('-'*20)
    test03()
