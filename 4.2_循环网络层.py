import torch
import torch.nn as nn


# 1. RNN 输入单个词
def test01():
    # 初始化RNN网络层
    # input_size: 输入句子的每个词的维度，比如： '我'，经过词用词嵌入层后，维度为128
    # hidden_size: 隐藏层的维度，比如：256,影响到最终输出的维度
    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 输入数据
    # 输入数据维度为：(seq_len, batch, input_size)
    # seq_len: 输入的词的个数，比如：'我'，'是'，'一个'，'程序员'，'。'，'我'，'喜欢'，'Python'，'。'，'我'，'喜欢'，'机器'，'学习'，'。'
    # batch: 批次数，比如：1
    input = torch.randn(8, 1, 128)
    # 初始化隐藏层
    hidden = torch.zeros(1, 1, 256)

    # 运行RNN网络层
    output, hidden = rnn(input, hidden)
    print(output.shape)
    print(hidden.shape)


# 2. RNN 输句子
def test02():
    # 初始化RNN网络层

    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 输入数据
    # input_size: （seq_len, batch, input_size）
    # 输入句子长度为 8 ，一次输入一个句子
    input = torch.randn(8, 1, 128)
    # 初始化隐藏层
    hidden = torch.zeros(1, 1, 256)

    # 运行RNN网络层
    output, hidden = rnn(input, hidden)
    print(output.shape)
    print(hidden.shape)

# 2. RNN 输入批次的数据
def test03():
    # 初始化RNN网络层

    rnn = nn.RNN(input_size=128, hidden_size=256)

    # 输入数据
    # input_size: （seq_len, batch, input_size）
    # 输入句子长度为 8 ，一次输入一个句子
    input = torch.randn(8, 16, 128)
    # 初始化隐藏层
    hidden = torch.zeros(1, 16, 256)

    # 运行RNN网络层
    output, hidden = rnn(input, hidden)
    print(output.shape)
    print(hidden.shape)



if __name__ == '__main__':
    test03()

