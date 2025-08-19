import torch

# 1、张量元素类型转换

def test ():
    #1.1 使用type进行类型转换
    data = torch.full((2,3),5)
    print(data.dtype)

    # 返回一个新的张量，类型转换  
    data = data.type(torch.DoubleTensor)

    print(data.dtype)
    
    data = data.type(torch.FloatTensor)
    print(data.dtype)

# 2、使用具体类型进行类型转换

def test02 ():
    data = torch.full((2,3),5)
    print(data.dtype)
     # 转换为double 类型
    data = data.double()
    print(data.dtype)

    data = data.short() # 转换为short类型
    date =data.long() # 转换为long类型
    date =data.float() # 转换为float类型
    print(data.dtype)


#     


if __name__ == '__main__':
    test()


