import torch
"""
注意：
    1. 梯度计算
       1. 默认情况下，torch.tensor()创建的tensor，其requires_grad属性为False，即不进行梯度计算
       2. 创建张量时，requires_grad属性为True，则张量会进行梯度计算
    2. 共享数据
      1. 创建张量时，requires_grad属性为True，则张量会共享数据，即x和y会共享数据
      2. 创建张量时，requires_grad属性为False，则张量不会共享数据，即x和y不会共享数据
      3. 创建张量时，requires_grad属性为None，则张量会共享数据，即x和y会共享数据
      4. 创建张量时，requires_grad属性为None，则张量不会共享数据，即x和y不会共享数据
 

"""
# 1. 演示下错误
def test01():
    x = torch.tensor([10,20],requires_grad=True,dtype=torch.float64 )
    # RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
    # print(x.numpy())
    # 以下是正确的
    
    print(x.detach().numpy())

# 2. 共享数据
def test02():
    x = torch.tensor([10,20],requires_grad=True,dtype=torch.float64 )
    y = x.detach()
    print(y.requires_grad)

    # 修改y
    y[0] = 100
    print(x)
    print(y)
    print(x.requires_grad)
    print(y.requires_grad)



if __name__ == '__main__':
    test02()

