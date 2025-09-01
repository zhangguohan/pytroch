import torch
import torch.nn as nn
import torch.optim as optim

# 1. 损失函数
def test01():
    # 初始化平方损失函数对象
    loss_fn = nn.MSELoss()
    # 该类内部重写 __call__ 方法, 所以对象可以当函数一样直接调用
    y_pred = torch.randn(3,5, requires_grad=True)
    y_true = torch.randn(3,5)
    loss = loss_fn(y_pred, y_true)
    print(loss)


# 2. 假设函数
def test02():
    # 输入特征数量必须要有10个, 输出特征数量必须要有5个
    model = nn.Linear(in_features=10, out_features=5)
    # 输入数据
    inputs = torch.randn(4,10)
    y_pred = model(inputs)
    print(y_pred.shape)


# 3. 优化函数
def test03():
    model = nn.Linear(in_features=10, out_features=5)
    # 优化方法：更新模型参数
    optimizer = optim.SGD(model.parameters(), lr=1e-2)
    # 在backward()方法中调用之前，需要进行梯度清零
    optimizer.zero_grad()
    ##此处省略了 backward() 方法,假设模型已经计算好梯度
    optimizer.step()








if __name__ == '__main__':
    test03()