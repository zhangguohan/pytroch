import torch
import torch.nn as nn


if __name__ == '__main__':
    # 输入的形状：{batch_size, channels, height, width}
    input = torch.randint(0, 10, [1, 2, 3, 3]).float()
    print(input)
    print('_'*20)

    # num_features: 输入的通道数
    # affine: 为False时，表示不带Gamma和Beta。两个学习参数
    # eps: 小常数，防止除零
    bn = nn.BatchNorm2d(num_features=2, affine=False, eps=1e-05)
    output = bn(input)
    print(output)
    print('_'*20)