import torch
import numpy as np

# 1. from_numpy 方法的使用
def test01():
    """
    将numpy数组转换成张量
    :return:
    """
    data_numpy = np.array([2, 3, 4])

    print(type(data_numpy))

    b = torch.from_numpy(data_numpy)
    print(type(b))
    print(b)

    # 默认认是共享内存
    data_numpy[0] = 100
    print(b)
    print(data_numpy)

# 2. torch.tensor() 方法的使用
def test02():
    """
    将numpy数组转换成张量
    :return:
    """
    data_numpy = np.array([2, 3, 4])
    data_tensor = torch.tensor(data_numpy)
    ## 使用torch.tensor()方法 is not shared memory
    #data_numpy[0] = 100
    data_tensor[0] = 100
    print(data_tensor)
    print(data_numpy)

if __name__ == '__main__':
    test02()