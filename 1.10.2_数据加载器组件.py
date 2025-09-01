import torch
from torch.utils.data  import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset


# 1. 数据类构建

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        idx = min(max(idx, 0), len(self.x) - 1)
        return self.x[idx], self.y[idx]
    
def test01():
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0),])

    dataset = SimpleDataset(x, y)
    print(len(dataset))
    print(dataset[0])

# 2. 数据加载器使用
def test02():
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0),])
    dataset = SimpleDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16,shuffle=True)
    for batch_x, batch_y in dataloader:
        print(batch_x, batch_y)
        break


# 3. 简单的数据类型构建方法
def test03():
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, [x.size(0),])
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=16,shuffle=True)
    for batch_x, batch_y in dataloader:
        print(batch_x, batch_y)
        break


if __name__ == '__main__':
    test02()
