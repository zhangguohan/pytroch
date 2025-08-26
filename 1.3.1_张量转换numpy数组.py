import torch
import numpy as np


# 1. 张量转换numpy数组
def test():
    a = torch.ones(5)
    print(type(a))
    b = a.numpy()
    print(type(b))
    a.add_(1)
    print(a)
    print(b)




# 2. 张量和numpy数组共享内存
def test02():
    a = torch.ones(5)
    print(type(a))
    b = a.numpy()
    print(type(b))
    # 修改张量元素,numpy数组也会被修改
    a.add_(1)
    print(a)
    print(b)


# 3. 使用copy函数实现不共享内存

def test03():
    a = torch.ones(5)
    print(type(a))
    # 转换为numpy数组，使用copy函数 避免inplace操作
    b = a.numpy().copy()
    print(type(b))
    # 创建新的张量,不会共享内存
    print(a)
    a.add_(1)
    print(b)
    print(a)





if __name__ == '__main__':
    test03()