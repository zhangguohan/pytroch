import torch

# 1.提取单值的张量 使用item()
def test01():
    t1 = torch.tensor(30)
    t2 = torch.tensor([30])
    t3 = torch.tensor([[30]])
    print(t1.shape)
    print(t2.shape)
    print(t3.shape)
    print(t1.item())
    # 注意 张量只存在一个元素，如果是单值的张量，则使用item()方法提取单值 多个元素张量则不能使用item()方法

if __name__ == '__main__':
    test01()
