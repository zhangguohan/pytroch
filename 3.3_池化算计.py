import torch
import torch.nn as nn

# 1.API 基本使用
def test01():
    inputs = torch.tensor([[0,1,2],[3,4,5],[6,7,8]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)
    # 1.最大池化
    # 输入数据的形状为(batch_size,channel,height,width)
    polling = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)


    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)


# 2. stride 步长
def test02():
    inputs = torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)
    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)
    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)

# 3. padding


def test03():
    inputs = torch.tensor([[0,1,2],[3,4,5],[6,7,8]]).float()
    inputs = inputs.unsqueeze(0).unsqueeze(0)
    print(inputs.shape)
    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2,stride=1,padding=1)    # 填充1
    outputs = polling(inputs)
    print(outputs.shape)
    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2,stride=1,padding=1)   # 填充1
    outputs = polling(inputs)
    print(outputs.shape)
    
# 多通道化
def test04():
    inputs = torch.tensor([[[0,1,2],[3,4,5],[6,7,8]],[[0,1,2],[3,4,5],[6,7,8]]]).float()
    inputs = inputs.unsqueeze(0)
    print(inputs.shape)
    # 1.最大池化
    polling = nn.MaxPool2d(kernel_size=2,stride=1,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)
    # 2.平均池化
    polling = nn.AvgPool2d(kernel_size=2,stride=1,padding=0)
    outputs = polling(inputs)
    print(outputs.shape)


if __name__ == '__main__':
    test04()
