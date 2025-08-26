import torch



def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4, 5])
    # 查看张量的形状
    print(data)
    print(data.shape, data.shape[0],data.shape[1])
    print(data.size(), data.size(0),data.size(1))


    # 修改张量的形状,注意需要确保修改后的形状和张量的元素数量一致
    data = data.reshape(2,10)
    print(data)

    # 可以使用-1, 表示任意大小
    data = data.reshape(5,-1)
    print(data)
    print(data.shape)

    data = data.reshape(-1,5)
    print(data)
    print(data.shape)

if __name__ == '__main__':
    test01()

