import torch


# 1.使用cuda 方法

def test():
    data = torch.tensor([10,20,30])
    print('存储设备',data.device)
    # 将张量移动到GPU
    data = data.cuda()
    print('存储设备',data.device)
    # 将张量移动到CPU
    data = data.cpu()
    print('存储设备',data.device)

                   
# 2.直接将张量创建在GPU上
def test02():
    data = torch.tensor([10,20,30],device='cuda:1')
    print('存储设备',data.device)
    # 将张量移动到CPU
    data = data.cpu()
    print('存储设备',data.device)

# 3. 使用to 方法

def test03():
    data = torch.tensor([10,20,30])
    print('存储设备',data.device)
    # 将张量移动到GPU
    data = data.to('cuda:0')
    print('存储设备',data.device)
    # 将张量移动到CPU
    data = data.to('cpu')
    print('存储设备',data.device)

# 4. 注意：存储在不同设备上的张量不能进行运算
def test04():
    data = torch.tensor([10,20,30],device='cuda:0')
    data1 = torch.tensor([10,20,30],device='cuda:1')
    data = data.to('cuda:1')
    data = data + data1
    print(data)


if __name__ == '__main__':

    test04()

