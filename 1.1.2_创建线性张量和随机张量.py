import torch
import numpy as np
#print(torch.__version__)
#print(torch.cuda.is_available())

# 1.创建线性张量

def test ():
    # 1.1 创建指定步长的张量
    # 第一参数指定起始值
    # 第二参数指定结束值
    # 第三参数指定步长
    # data = torch.arange(1,10,2)
    # print(data)
    # print(data.dtype)
    # print(data.shape)

    # 1.2 在指定区间指定元素个数
    # 第一个参数指定起始值
    # 第二个参数指定结束值
    # 第三个参数指定元素个数

    data = torch.linspace(1,10,100)
    print(data)
    print(data.dtype)
    print(data.shape)


# 2.创建随机张量

def test2 ():
    # 2.1 创建随机张量
    data = torch.rand(2,3)
    print(data)
    print(data.dtype)
    print(data.shape)


    # 2.2 创建固定随机数的张量
    # 第一个参数指定随机数的范围
    # 第二个参数指定张量的形状
    # 创建一个2行3列的矩阵，矩阵中的元素都是1-10的随机数
    data = torch.randint(1,10,(2,3))
    print(data)
    # 2.3 创建固定随机数的张量
    # 固定随机数种子
    torch.manual_seed(0)
    data = torch.randint(10,(2,3))
    print("随机数种子",torch.initial_seed())
    print(data)
     




if __name__ == '__main__':
    test()
