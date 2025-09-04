import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import time
import os
"""
网络性能调优
1.对输入数据进行标准化
2.调整优化方法
3.调整学习率
4.增加批量归一化层
5.增加网络层数，神经元数
6.增加训练轮决
7.等等
"""
# 1.构建数据集
"""
def create_dataset():
    # 读取数据集
    data = pd.read_csv('data/train.csv')
    # 将特征值和目标值分别取出来
    x, y = data[data.columns[1:-1]], data[data.columns[-1]]
    # x 的数据类型是 float64
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.88, random_state=88, stratify=y)
    
    # 对输入数据进行标准化 (均值为0，标准差为1)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    # 避免除零错误，将标准差为0的特征设置为1
    std[std == 0] = 1
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # 构建pytorch的数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.tensor(y_train.values)) 
    test_dataset = TensorDataset(torch.from_numpy(x_test.values), torch.tensor(y_test.values))
    # 返回数据集:训练对象，测试对象，特征维度
    return train_dataset, test_dataset, x_train.shape[1],len(np.unique(y))
train_dataset, test_dataset, input_dim, num_classes = create_dataset()
"""
# ... existing code ...
def create_dataset():
    # 读取数据集
    data = pd.read_csv('data/train.csv')
    # 将特征值和目标值分别取出来
    x, y = data[data.columns[1:-1]], data[data.columns[-1]]
    # x 的数据类型是 float64
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.88, random_state=88, stratify=y)
    
    # 对输入数据进行标准化 (均值为0，标准差为1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # 构建pytorch的数据集
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.values)) 
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test.values))
    # 返回数据集:训练对象，测试对象，特征维度
    return train_dataset, test_dataset, x_train.shape[1],len(np.unique(y))
train_dataset, test_dataset, input_dim, num_classes = create_dataset()
# ... existing code ...


# 2.构建分类网络模型
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        # 调用父类的初始化方法
        super(PhonePriceModel, self).__init__()
        # 定义网络结构
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 256)
        #  定义输出层
        self.linear5 = nn.Linear(256, num_classes)
    #  定义激活函数    
    def _activation(self, x):
        return torch.sigmoid(x)
    # 定义前向传播
    def forward(self, x):
        x = self._activation(self.linear1(x))
        x = self._activation(self.linear2(x))
        x = self._activation(self.linear3(x))
        x = self._activation(self.linear4(x))
        output = self.linear5(x)
        return output
# 3.训练
def train():
    # 固定随机数种子
    torch.manual_seed(0)
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化网络模型并移动到GPU
    model = PhonePriceModel(input_dim, num_classes).to(device)
    # 定义损失函数 会首先对数据进行softmax,再志然进行交叉熵计算
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # 训练轮数,表示将所有数据集训练多少轮
    num_epochs = 500
    
    # 初始化数据加载器
    data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    for epoch in range(num_epochs):
        # 训练时间
        start_time = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 1
        # 预测正确样本数量
        correct = 0

        for x, y in data_loader:
            # 将数据移动到GPU
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

            # 累计总样本数量
            total_num += len(y)
            # 累计总损失
            total_loss += loss.item() * len(y)
            
            # 累计预测正确的样本数量
            y_pred = torch.argmax(output, dim=-1)
            correct += (y_pred == y).sum().item()
            
        print('epoch: %d, loss: %.4f, acc: %.4f, time: %.4f' %
               (epoch + 1, 
                total_loss/total_num,
                correct/total_num,
                time.time()-start_time))
    
    save_path = 'model/phone_price_model.pt'
    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存模型
    torch.save(model.state_dict(), save_path)



# 4.测试
def test():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建数据加载器
    data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 加载模型
    model = PhonePriceModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load('model/phone_price_model.pt'))
    # 确保模型在评估模式
    model.eval()
    
    # 计算准确率
    total_num = 0
    correct = 0
    for x, y in data_loader:
        # 将数据移动到GPU
        x, y = x.to(device), y.to(device)
        # 将数据送入网络
        output = model(x)
        # 得到预测结果
        y_pred = torch.argmax(output, dim=-1)
        total_num += len(y)
        correct += (y_pred == y).sum().item()
    print('acc: %.4f' % (correct/len(test_dataset)))
        

if __name__ == '__main__':
     #train()
     test()