import torch


# 1. 控制梯度计算
def test01():
    x = torch.tensor(10, requires_grad=True,dtype=torch.float64)
    print(x.requires_grad)
    # 1.第一种方法
    with torch.no_grad():
        y = x ** 2
    
    print(y.requires_grad)
    # 2.第二种方法
    @torch.no_grad()
    def fun(x):
        y = x ** 2
        return y
    y = fun(x)
    print(y.requires_grad)

    # 3.第三种方法,全局关闭
    torch.set_grad_enabled(False)
    y = x ** 2
    print(y.requires_grad)


# 2. 累计梯梯度和梯度清零

def test02():
    x = torch.tensor([10,20,30,40],requires_grad=True,dtype=torch.float64)
    # 当重新计算时，梯度值会累加，需要清零
    # 希望每次计算时，梯度值清零
    for i in range(10):
       # 对输入变量进行计算
       f1 = x ** 2 + 20
       # 将向量转换为张量     
       f2 = f1.mean()
       # 清零
       if f2.grad is not None:
           f2.grad.data.zero_()
       # 自动微分
       f2.backward()
       # 打印梯度值
       print(x.grad)


# 3. 梯度下降优化

def test03():
    x = torch.tensor(10,requires_grad=True,dtype=torch.float64)
    for i in range(5000):
        # 对输入变量进行计算
        f = x ** 2 
        # 自动微分
        f.backward()
        # 梯度下降
        x.data = x.data - 0.001 * x.grad
        # 清零
        x.grad.data.zero_()
        # 打印结果
        print(x.data)

if __name__ == '__main__':
    test03()