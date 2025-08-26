import torch

# 1. 布尔索引
def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4, 5])
    print(data)
    print('_' * 30 )
    # 1. 获取大于5的元素
    print(data > 5)
    print(data[data > 5])
    print(data[data > 5].shape)
    print(data[data > 5].view(-1))
    print(data[data > 5].view(-1).shape)
    # 2. 获取第2列中，大于5的元素
    print(data[:, 2][data[:, 2] > 5])
    print(data[:,2][data[:,2] > 5].shape)
    print('_' * 30 )
    # 3. 返回第2行元素大于5的所有列元素
    print(data[:, data[1] > 5])



# 2. 多维索引

def test02():
    torch.manual_seed(0)
    data = torch.randint(0,10,[3,4, 5])
    print(data)
    # 按照第0个维度选中第0个维度的元素，4行5列元素
    print('_' * 30 )
    print(data[0,1])
    print(data[0,1,2])
    print(data[0,1,2].shape)
    # 按照第1个维度选中第0元素    
    print(data[:,0,:])
    print('_' * 30 )



if __name__ == '__main__':
    test02()


