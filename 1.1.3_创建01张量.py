import torch

def test():

# 1创建全零张量

# 1.1 创建指定形状的全零张量
    data = torch.zeros(2,3)
    print(data)
# 1.2 根据其它形态创建全0张量
    data = torch.zeros_like(data)
    print(data)

def test02():
    # 1.2 创建全1张量
    data = torch.ones(2,3)
    print(data)
     # 1.3 根据其它形态创建全1张量
    data = torch.ones_like(data)
    print(data)



# 3 创建全为指定值的张量
def test03():
  
    # 3.1 创建指定值张量
    # 第一个参数为张量的形状
    # 第二个参数为张量的值
    data = torch.full((2,3),5)
    print(data)

    # 3.2 根据其它张量创建指定值张量
    data = torch.full_like(data,30)
    print(data)

if __name__ == '__main__':
    #test()
    #test02()
    test03()
