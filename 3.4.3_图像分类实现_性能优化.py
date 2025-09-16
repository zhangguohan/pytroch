import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import time
import os

"""

1、增加卷积核输出通道数
2、增加卷积核数量
3、增加全连接层数量
4、增加全连接层神经元数量
5、调整激活函数
6、调整学习率

"""


# 创建图像分类类
class ImageClassification(nn.Module):
    def __init__(self):
        # 调用父类初始化方法
        super(ImageClassification, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 卷积层
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        
        # 动态计算展平维度
        flatten_size = x.size(1) * x.size(2) * x.size(3)
        # 展平
        x = x.view(-1, flatten_size)

        # 全连接层
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout(x)
        x = torch.relu(self.linear3(x))
        x = self.output(x)
        return x
    
# 编写训练函数    
def train():
    try:
        # 加载数据集
        cifar10 = CIFAR10(root='./data', train=True, transform=Compose([ToTensor()]), download=True)

        # 初始化网络
        model = ImageClassification()
        
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()

        # 定义优化器
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # 训练轮数
        epochs = 10

        for epoch in range(epochs):

            # 初始化数据加载器
            data_loader = DataLoader(cifar10, batch_size=32, shuffle=True)
            # 样本数量
            num_samples = 0
            # 损失总和
            total_loss = 0.0
            # 开始时间
            start_time = time.time()

            # 正确样本数量
            correct = 0
            for x, y in data_loader:
                # 将数据送入网络
                output = model(x)
                # 计算损失
                loss = criterion(output, y)
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # 统计信息
                total_loss += loss.item() * len(y)
                num_samples += len(y)
                # 计算准确率
                correct += (torch.argmax(output, dim=-1) == y).sum().item()
                
            print('epoch: %d, loss: %.4f, acc: %.4f, time: %.2f' % (epoch + 1, total_loss/num_samples, correct/num_samples, time.time()-start_time))
            
        # 确保模型保存目录存在
        os.makedirs('model', exist_ok=True)
        # 保存模型
        torch.save(model.state_dict(), 'model/cifar10-opt.pt')
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        raise


# 测试函数
def test():
    try:
        # 加载数据集
        cifar10 = CIFAR10(root='./data', train=False, transform=Compose([ToTensor()]), download=True)
        # 初始化数据加载器
        data_loader = DataLoader(cifar10, batch_size=32, shuffle=False)
        # 初始化网络
        model = ImageClassification()
        model.load_state_dict(torch.load('model/cifar10-opt.pt'))
        # 模型有两种状，一个用于训练，一个用于测试
        model.eval() # 测试模式

        total_correct = 0
        total_samples = 0
        
        # 在测试时禁用梯度计算
        with torch.no_grad():
            for x, y in data_loader:
                # 将数据送入网络
                output = model(x)
                # 获取预测结果
                total_correct += (torch.argmax(output, dim=-1) == y).sum().item()
                total_samples += len(y)
        print('acc: %.4f' % (total_correct/total_samples))
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        raise


if __name__ == '__main__':
    test()