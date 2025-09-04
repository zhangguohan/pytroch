import torch
import matplotlib.pyplot as plt


# 1.没有指数加权平均
def test01():
    # 固定随机数种子
    torch.manual_seed(0)
    # 生成数据
    temperature = torch.randn(size=[30,]) * 10
    
    # 绘制温度曲线 
    days = torch.arange(1, 31)
    plt.plot(days, temperature,'o-r')
    plt.savefig('result.png')

# 2.指数加权平均
# ... existing code ...
def test02(beta=0.9):
    # 1.生成数据
    torch.manual_seed(0)
    temperature = torch.randn(size=[30,]) * 10
    days = torch.arange(1, 31)
    # 2.计算指数加权平均
    exp_weight_avg = []
    for idx, temp in enumerate(temperature,1):
        if idx == 1:
            exp_weight_avg.append(temp)
            continue
        else:
            new_temp = exp_weight_avg[idx-2] * beta + temp * (1-beta) * temp
            exp_weight_avg.append(new_temp)
    plt.plot(days, exp_weight_avg,'o-r')
    plt.savefig('result029.png')
# ... existing code ...


if __name__ == '__main__':
    test02(0.5)
    