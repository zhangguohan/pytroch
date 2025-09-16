import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
import time

# 创建图像分类类
class ImageClassification(nn.Module):
    def __init__(self):
        # 调用父类初始化方法
        super(ImageClassification, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 6, stride=1, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, stride=1, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)
    def forward(self, x):
        # 卷积层
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 400)

        # 全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output(x)
        return x
    def test(self, x):
        # 卷积层
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # 展平 (修复维度)
        x = x.view(-1, 400)

        # 全连接层
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.output(x)
    
# 编写训练函数    
def train():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    cifar10 = CIFAR10(root='./data', train=True, transform=Compose([ToTensor()]), download=True)

    # 初始化网络并移动到设备
    model = ImageClassification().to(device)
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练轮数
    epochs = 20

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
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)
            
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
            correct += (torch.argmax(output, dim=1) == y).sum().item()
            total_loss += loss.item() * len(y)
            num_samples += len(y)
        print('epoch: %d, loss: %.4f, acc: %.4f, time: %.2f' % (epoch + 1, total_loss/num_samples, correct/num_samples, time.time()-start_time ))        
    # 保存模型
    torch.save(model.state_dict(), 'model/cifar10.pt')


if __name__ == '__main__':
    train()