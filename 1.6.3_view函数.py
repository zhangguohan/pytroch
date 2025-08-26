"""
1. 基本作用

view() 用来 重新调整张量的形状 (reshape)，但是不会改变数据本身。

⚠️ 前提：调用 view() 的张量必须是 连续内存 (contiguous) 的，如果不是，需要先 .contiguous()

"""

import torch

# 1.veiw() 函数的使用

def test01():
    torch.manual_seed(0)
    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    data = data.view(3, 2)
    print(data.shape)
    # is_contiguous() 判断张量是否是连续的
    print(data.is_contiguous())


# 2. view() 函数的注意点
def test02():
    # 当张量经过 transpose() 函数后，张量不再是连续的，内存布局会改变，此时再调用 view() 函数就会报错
    torch.manual_seed(0)
    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print('是否连续',data.is_contiguous())
    data = data.transpose(0, 1)
    # is_contiguous() 判断张量是否是连续的
    print('是否连续',data.is_contiguous())

    data = data.contiguous().view(2, 3)
    print(data)

if __name__ == '__main__':
    test02()
