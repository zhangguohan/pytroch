"""
torch.add(x,y)        # 加法
torch.sub(x,y)        # 减法
torch.mul(x,y)        # 逐元素乘
torch.div(x,y)        # 逐元素除
torch.matmul(x,y)     # 矩阵乘法
torch.mm(x,y)         # 矩阵乘法 (2D)
torch.bmm(x,y)        # batch 矩阵乘法
x @ y                 # 矩阵乘法符号

"""

import torch


# 1. 均值
def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求均值
    print(data.mean())

# 2. 求和
def test02():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求和
    print(data.sum())

# 3. 乘积    

def test03():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求乘积
    print(data.prod())
    print(data.prod(0))

# 4. 平方
def test04():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求平方
    print(data.pow(2))

# 5. 平方根
def test05():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求平方根
    print(data.sqrt())

# 6. 绝对值    
def test06():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求绝对值
    print(data.abs())

# 7. 对数
def test07():
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3]).double()
    print(data)
    # 默认对所有元素求对数
    print(data.log())
    print(data.log2())
    print(data.log10())

if __name__ == '__main__':
    test07()
