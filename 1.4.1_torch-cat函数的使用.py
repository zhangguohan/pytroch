import torch
# 1.使用cat函数
def test():
    torch.manual_seed(0)
    data1 = torch.randint(0,10,[3, 4, 5])
    data2 = torch.randint(0,10,[3, 4, 5])
    print(data1.shape)
    print(data2.shape)

    # 1. 按照0维度进行拼接
    new_data = torch.cat([data1, data2], dim=0)
    print(new_data.shape)

    # 2. 按照1维度进行拼接
    new_data = torch.cat([data1, data2], dim=1)
    print(new_data.shape)
    # 3. 按照2维度进行拼接
    new_data = torch.cat([data1, data2], dim=2)
    print(new_data.shape)





if __name__ == '__main__':
    test()