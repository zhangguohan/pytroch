import torch
import numpy as np
#print(torch.__version__)
#print(torch.cuda.is_available())

#1.根据已有的数据创建张量
def test01():
    # 1.1 创建标量
    a = torch.tensor(10)
    print(a)

    # 1.2 使用numpy数组创建张量
    data = np.random.randn(3,3)
    data = torch.tensor(data)
    print(data)

    # 1.3 使用list 列表创建张量
    data = [[1.,2.,3.],[4.,5.,6.]]
    data = torch.tensor(data)
    print(data)


#2.创建指定形状的张量
def test02():
    # 2.1 创建2行3列的张量
    data = torch.Tensor(2,3)
    print(data)
    #2.2 可以创建指定值的张量
    # 注意：传递列表
    data = torch.Tensor([2,3])
    print(data)
    data = torch.Tensor([19])
    print(data)

#3.创建指定类型的张量
def test03():
    # 3.1 创建一个全0的矩阵
    data = torch.zeros(2,3)
    print(data)
    # 3.2 创建一个全1的矩阵
    data = torch.ones(2,3)
    print(data)
    # 3.3 创建一个全0的矩阵，并指定数据类型
    data = torch.zeros(2,3,dtype=torch.int)
    print(data)
    # 3.4 创建一个int 32类型的张量
    data = torch.IntTensor(2,3)
    print(data)
    # 3.5 创建一个float 32类型的张量
    data = torch.FloatTensor(2,3)
    print(data)
    # 3.6 创建一个float 64类型的张量
    data = torch.DoubleTensor(2,3)
    print(data)
    # 3.7 创建一个随机矩阵，并指定数据类型
    data = torch.rand(2,3,dtype=torch.float64)
    print(data)

if __name__ == '__main__':
    #test01()
    #test02()
    test03()
