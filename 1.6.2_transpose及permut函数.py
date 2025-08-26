"""
1.transpose(dim0, dim1)

功能：只交换两个维度的位置。

参数：需要指定要交换的两个维度索引 dim0 和 dim1。

常用场景：矩阵转置（二维张量），或者在高维张量里交换两个维度。

2. permute(dims)

功能：可以同时改变多个维度的顺序。

参数：传入一个包含所有维度索引的新顺序（tuple 或 list）。

常用场景：对高维张量进行复杂维度重排。

"""
import torch

def test01():
        # 创建一个形状为 (2, 3, 4) 的随机张量
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3])

    print("原始张量:")
    print(data)
    print(data.shape)  # 输出: torch.Size([2, 3])

    # 使用 transpose 函数交换前两个维度 (dim0 和 dim1)
    transposed_tensor = data.transpose(0, 1)

    print("\n转置后的张量1:")
    print(transposed_tensor)
    print(transposed_tensor.shape)  # 输出: torch.Size([3, 2])



def test02():

# 创建一个形状为 (2, 3, 4) 的随机张量
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3])

    print("原始张量2:")
    print(data)
    print(data.shape)  # 输出: torch.Size([2, 3])

    # 使用 permute 函数将维度顺序改为 (1, 0)
    permuted_tensor = data.permute(1, 0) 

    print("\n排列后的张量:")
    print(permuted_tensor)
    print(permuted_tensor.shape)  # 输出: torch.Size([3, 2])

if __name__ == '__main__':
    test01()
    test02()
