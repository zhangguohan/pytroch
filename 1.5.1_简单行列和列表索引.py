import torch

# 1.简单行列
def test01():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4, 5])
    print(data)
    #print('_' * 30 )
    # 1. 获取第一行元素
    #print(data[0])
    #print('_' * 30 )
    # 2. 获取指定列元素
    # 冒号前面表示行，逗号后面表示列
    print(data[:,:])
    # 3. 获取指定列元素
    print(data[:,2])
    # 5.获取提定前3行的第3列元素
    print(data[:3,2] )
    # 4. 获取指定位置元素
    print(data[1,4])
    print('_' * 30 )
    # 6. 获取前三行的前2列元素
    print(data[:3,:2])



# 2.列表索引

def test02():
    torch.manual_seed(0)
    data = torch.randint(0,10,[4, 5])
    print(data)
    print(data[0])
    # 1. 获取0，2，3行的，0，1，2列元素
    data = data[[0,2,3],[0,1,3]]
    print(data)
    print('_' * 30 )







if __name__ == '__main__':
    test02()

