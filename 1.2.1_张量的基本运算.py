import torch

# 1. 不修源数据的计算
def test ():
    # 第一个参数：开始值
    # 第二个参数：结束值
    # 第三个参数：张量的形状
    data = torch.randint(1,10,(2,3))
    print(data)
    # 计算完全之后，会返回一个新的张量
    data = data.add(10)
    print(data)
    # data.sub(10) // 减法
    # data.mul(10) // 乘法 
    # data.div(10) // 除法
# 2 修改源数据的计算

def test02 ():
    data = torch.randint(1,10,(2,3))
    print(data)
    # 源数据进行修改,并不需要返回新的张量
    data.add_(10)
    print(data)
    # data.sub_(10) // 减法
    # data.mul_(10) // 乘法 
    # data.div_(10) // 除法


if __name__ == '__main__':
    test02()

