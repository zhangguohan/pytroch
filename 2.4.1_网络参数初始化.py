import torch
import torch.nn as nn

# 1. 均匀分布初始化
def test():
    # 输入数据的特征维度是5, 输出数据的特征维度是3
    linear = nn.Linear(5,3)
    nn.init.uniform_(linear.weight)
    print(linear.weight)
# 2. 固定初始化
def test02():
    linear = nn.Linear(5,3)
    nn.init.constant_(linear.weight, 0.0)
    print(linear.weight)

# 3. 全零初始化
def test03():
    linear = nn.Linear(5,3)
    nn.init.zeros_(linear.weight)
    print(linear.weight)

# 4. 全1初始化
def test04():
    linear = nn.Linear(5,3)
    nn.init.ones_(linear.weight)
    print(linear.weight)

# 5. 正态分布初始化
def test05():
    linear = nn.Linear(5,3)
    nn.init.normal_(linear.weight, mean=0.0, std=1.0)
    print(linear.weight)
# 6. kaiming初始化

def test06():
    # 正态分布 kaiming_normal_
    linear = nn.Linear(5,3)
    nn.init.kaiming_normal_(linear.weight)
    print(linear.weight)
    # 均匀分布 kaiming_uniform_
    linear = nn.Linear(5,3)
    nn.init.kaiming_uniform_(linear.weight)
    print(linear.weight)
# 7. xavier初始化
def test07():
    # 正态分布 xavier_normal_
    linear = nn.Linear(5,3)
    nn.init.xavier_normal_(linear.weight)
    print(linear.weight)
    # 均匀分布 kaining_uniform_
    linear = nn.Linear(5,3)
    nn.init.xavier_uniform_(linear.weight)
    print(linear.weight)



if __name__ == '__main__':
    test06()