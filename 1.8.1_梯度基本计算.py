import torch

# 1. 标量的梯度计算
# y = x ** 2 + 20

def test01():
    # 对于需要求梯度的张量，需要设置requires_grad=True
    x = torch.tensor(10, requires_grad=True,dtype=torch.float64)
    # 对X的中间进行计算
    f = x ** 2 + 20
    # 自动微分
    f.backward()
    print(f)
    print(x.grad)


# 2. 向量的梯度计算
# y = x ** 2 + 20

def test02():
    x = torch.tensor([10,20,30,40], requires_grad=True,dtype=torch.float64)
    # 定义变量的计算过程
    y1 = x ** 2 + 20
    # 注意：自动微分分计算，需要调用mean()
    y2= y1.mean()
    # 自动微分
    y2.backward() ## 1/4 *y1 ==> 1/4 *2x
    #print(y1)

    # 打印梯度值 
    print(x.grad)   

# 3. 多标量的梯度计算
def test03():
   x1 = torch.tensor(10, requires_grad=True,dtype=torch.float64)
   x2 = torch.tensor(20, requires_grad=True,dtype=torch.float64)
   # 中间计算过程
   y = x1 ** 2 + x2 ** 2 + 20
   # 自动微分
   y.backward()
   # 打印梯度值
   print(x1.grad)
   print(x2.grad)

# 4. 多向量的梯度计算
def test04():   
    x1 = torch.tensor([10,20,30,40], requires_grad=True,dtype=torch.float64)
    x2 = torch.tensor([10,20,30,40], requires_grad=True,dtype=torch.float64)
    # 中间计算过程
    y = x1 ** 2 + x2 ** 2 + 20
    y = y.sum()

    # 自动微分
    y.backward()
    # 打印梯度值
    print(x1.grad)
    print(x2.grad)
   

if __name__ == '__main__':
    test04()
