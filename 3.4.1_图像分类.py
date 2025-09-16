from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torch.utils.data import DataLoader


# 1. 数据的基本信息
def test01():
    # 1.创建数据集
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=Compose([ToTensor()]))
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=Compose([ToTensor()]))
    # 数据集的数量
    print('训练集数量:',len(train_dataset))
    print('测试集数量:',len(val_dataset))

    # 数据集的形状
    print('训练集数据形状:',train_dataset[0][0].shape)
    print('测试集数据形状:',val_dataset[0][0].shape)

    # 数据集的类别
    print('训练集类别:',train_dataset.classes)

# 2.数据加载器构建
def test02():
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=Compose([ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for x, y in train_loader:
        print('数据形状:',x.shape)
        print('标签形状:',y.shape)
        break


if __name__ == '__main__':
    test02()
