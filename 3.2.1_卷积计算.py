import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def show_img(img):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('data/imgtr.png')

# 1.单个卷积核
def test01():
    # 读取图片(1222, 651, 3) -->(H, W, C)
    img = plt.imread('data/钢铁侠.jpg')
    print(img.shape)
    # 构建卷积核
    # in_channels: 输入图像的通道数
    # out_channels: 指的是当输入一个图像后，产生的图像通道数
    # kernel_size: 卷积核的大小
    # stride: 卷积核的步长
    # padding: 卷积核的填充
    # 构建卷积核

    conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
    # 额外注意，卷积层对输入的数据有形状要求（Batch_size, Channels, Height, Width）
    # 将图片的形状（H,W,C）转换成(C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)

    # 将（C, H, W)转换成(Batch_size, C, H, W)）
    new_img = img.unsqueeze(0)
    print(new_img.shape)

    # 将数据传入卷积核中进积核进行卷积
    new_img = new_img.float()
    output = conv(new_img)
    print(output.shape)
    # 将 (Batch_size, C, H, W) --> (H, W, C)
    show_img(output.squeeze(0).permute(1, 2, 0).detach().numpy())




# ... existing code ...

# 2.多个卷积核
def test02():
    # 读取图片(1222, 651, 3) -->(H, W, C)
    img = plt.imread('data/钢铁侠.jpg')
    print(img.shape)
    # 构建卷积核
    # in_channels: 输入图像的通道数
    # out_channels: 指的是当输入一个图像后，产生的图像通道数
    # kernel_size: 卷积核的大小
    # stride: 卷积核的步长
    # padding: 卷积核的填充
    # 构建卷积核

    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    # 额外注意，卷积层对输入的数据有形状要求（Batch_size, Channels, Height, Width）
    # 将图片的形状（H,W,C）转换成(C, H, W)
    img = torch.tensor(img).permute(2, 0, 1)
    print(img.shape)
    # 将（C, H, W)转换成(Batch_size, C, H, W)）
    new_img = img.unsqueeze(0)
    print(new_img.shape)
    # 将数据传入卷积核中进积核进行卷积
    new_img = new_img.float()
    output = conv(new_img)
    print(output.shape)
    
    # 显示多个卷积核生成的3个图片
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        # 将 (Batch_size, C, H, W) --> (H, W) 并显示每个通道
        channel_img = output.squeeze(0)[i].detach().numpy()
        axes[i].imshow(channel_img, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i+1}')
    plt.tight_layout()
    plt.savefig('data/multi_kernel_result.png')
   

# ... existing code ...
   


if __name__ == '__main__':
    test02()