import torch

# 1. 使用@运算符
def test01 ():
    # 形状为 3 * 2
    data1 = torch.tensor([[1,2],
                          [3,4],
                          [5,6]])
    # 形状为 2 * 1
    data2 = torch.tensor([[5],[6]])
    print(data1 @ data2)
# 2. 使用 mm 函数
def test02 ():
    # 对于输入是二维的张量相当于mm 运算
    # 形状为 3 * 2
    data1 = torch.tensor([[1,2],
                          [3,4],
                          [5,6]])
    # 形状为 2 * 1
    data2 = torch.tensor([[5],[6]])

    print(torch.mm(data1, data2))


# 3. 使用 bmm 函数
def test03 ():
    # 对于输入的是三维张量 相当于 bmm 运算
    # 第一维为批量维
    # 第二维为矩阵的行
    # 第三维为矩阵的列
    data1 = torch.randn(3,4,5)
    data2 = torch.randn(3,5,8)
    data = torch.bmm(data1, data2)
    print(data.shape)


# 4. 使用 matmul 函数
def test04 ():
    # 第一维为批量维
    # 第二维为矩阵的行
    # 第三维为矩阵的列
    # 对二维矩阵进行矩阵乘积
    data1 = torch.randn(3,4)
    data2 = torch.randn(3,5)
    data = torch.matmul(data1, data2)
    print(data.shape)

    # 对三维矩阵进行矩阵乘积
    data1 = torch.randn(3,4,5)
    data2 = torch.randn(3,5,8)
    data = torch.matmul(data1, data2)
    print(data.shape)
    # 对三维矩阵进行矩阵乘积
    data1 = torch.randn(3,4,5)
    data2 = torch.randn(5,8)
    data = torch.matmul(data1, data2)
    print(data.shape)

if __name__ == '__main__':
        test04()
