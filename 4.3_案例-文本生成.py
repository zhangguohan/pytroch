import re  # 导入正则表达式模块，用于文本清洗和处理
import time
import jieba  # 导入jieba库，用于中文分词
import torch  # 导入 PyTorch 库，用于深度学习
import torch.nn as nn  # 从 PyTorch 中导入神经网络模块
import torch.nn.functional as F  # 从 PyTorch 中导入功能模块，用于实现一些常见的神经网络操作
import torch.optim as optim  # 导入 PyTorch 的优化器模块
from torch.utils.data import DataLoader, Dataset  # 从 PyTorch 中导入 DataLoader 类，用于批量数据加载

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# 构建词典
def build_vocab():
    fname = 'data/jaychou_lyrics.txt'
    # 1.文本数据清洗
    # 初始化一个列表，用于存储清洗后的句子
    clean_sentences = []
    for line in open(fname, 'r', encoding='utf-8'):  # 打开歌词文件，逐行读取，指定编码为UTF-8
        line = line.replace('〖韩语Rap译文〗', '')  # 去除特定的文本标记
        # TODO 使用正则表达式去除无效字符
        # 去除 除了 中文、英文、数字、部分标点符号外 的其他字符
        line = re.sub(r'[^\u4e00-\u9fa5 a-zA-Z0-9!?,]', '', line)
        # 连续空格替换成1个
        line = re.sub(r'[ ]{2,}', '', line)  # 替换连续的空格为单个空格
        # 去除两侧空格、换行
        line = line.strip()  # 去除行首和行尾的空白字符和换行符
        if len(line) <= 1:  # 如果行的长度小于等于1，则跳过该行
            continue
        # 去除重复行
        if line not in clean_sentences:  # 如果行不在clean_sentences中，则添加
            clean_sentences.append(line)  # 将清洗后的行添加到clean_sentences列表
    print('共有%d行歌词' % len(clean_sentences))

    # 2.分词
    all_words = []
    index_to_word = [] # 重要的词表： 索引到单词的映射
    for line in clean_sentences:
        words = jieba.lcut(line)
        # 便于后面我们将句子转换成索引，因此需要将单词转换为索引
        all_words.append(words)
        for word in words:
            if word not in index_to_word:
                index_to_word.append(word)
    # 重要词表：词到索引的映射
    word_to_index = {word: index for index, word in enumerate(index_to_word)}   
    
   # 3. 构建语料索引
    word_count = len(index_to_word)
    corpus_idx = []  # 初始化一个列表，用于存储整个语料的索引表示
    for sentence in all_words:  # 遍历每个分词后的句子
        temp = []  # 初始化一个临时列表，用于存储句子的索引
        for word in sentence:  # 遍历句子中的每个词
            temp.append(word_to_index[word])  # 将词转换为索引并添加到临时列表中
        # 在每行歌词之间添加空格隔开
        temp.append(word_to_index[' '])  # 在每个句子末尾添加空格的索引作为分隔符
        # print("temp: ", temp)
        # TODO extend()是逐个添加, 区别于extend()
        corpus_idx.extend(temp)  # 将句子的索引表示添加到corpus_idx列表中
        # print("corpus_idx: ", corpus_idx)
    # TODO 返回构建的词汇表、索引映射、词数、语料索引
    return index_to_word, word_to_index, word_count, corpus_idx



# 处理歌词数据集
class LyricsDataset(Dataset):
    def __init__(self, corpus_idx, num_chars):  # 初始化方法，构造函数
        # 语料数据
        self.corpus_idx = corpus_idx  # 将传入的语料索引列表存储为类的属性
        # 语料长度
        self.num_chars = num_chars  # 将每个输入序列的长度（字符数）存储为类的属性
        # 词的数量
        self.word_count = len(self.corpus_idx)  # 计算语料索引列表的长度，即词汇总数
        # 句子数量
        self.number = self.word_count // self.num_chars  # 计算可以提取的样本序列的数量，每个样本长度为 num_chars

    def __len__(self):  # 定义返回数据集大小的方法
        return self.number  # 返回可以提取的样本序列数量

    # TODO 后续通过DataLoader()加载这里的数据
    def __getitem__(self, idx):  # 定义获取数据集中某个样本的方法
        # 修正索引值到: [0, self.word_count - 1]
        start = min(max(idx, 0), self.word_count - self.num_chars - 2)  # 限制起始索引，确保有效的样本提取范围
        x = self.corpus_idx[start: start + self.num_chars]  # 获取从起始索引开始的 num_chars 长度的序列作为输入
        y = self.corpus_idx[start + 1: start + 1 + self.num_chars]  # 获取从起始索引+1开始的 num_chars 长度的序列作为目标
        return torch.tensor(x), torch.tensor(y)  # 返回输入和目标序列作为张量



# 定义一个文本生成模型类，继承自 nn.Module
class TextGenerator(nn.Module):
    # 初始化方法，构造函数
    def __init__(self, vocab_size):
        print("vocab_size: ", vocab_size)
        # 调用父类（nn.Module）的构造函数
        super(TextGenerator, self).__init__()
        # TODO 定义新变量ebd、rnn、out
        # 初始化词嵌入层，将词汇表中的每个词映射到一个128维的向量
        self.ebd = nn.Embedding(vocab_size, 128)
        # 初始化循环神经网络层，输入和输出都是128维，层数为1
        self.rnn = nn.RNN(128, 128, 1)
        # 初始化线性输出层，将RNN的输出映射到词汇表的大小
        # TODO 输出是需要转为词汇, 所以vocab_size多大, 输出就是多大的
        self.out = nn.Linear(128, vocab_size)

    # 定义前向传播方法
    def forward(self, inputs, hidden):
        # 将输入的词索引转换为嵌入向量，输出维度为 (1, 5, 128)
        embed = self.ebd(inputs).to(device)
        # 对嵌入向量进行dropout正则化，防止过拟合，概率为0.2
        embed = F.dropout(embed, p=0.2)
        # 将嵌入向量的维度从 (1, 5, 128) 转置为 (5, 1, 128) 以匹配RNN的输入要求
        # TODO 这里会调用rnn.py的forward()方法
        output, hidden = self.rnn(embed.transpose(0, 1), hidden)  # embed.transpose(0, 1)调整张量维度
        # 对RNN的输出进行dropout正则化，防止过拟合，概率为0.2
        embed = F.dropout(output, p=0.2)
        # 将RNN的输出维度从 (5, 1, 128) 压缩为 (5, 128)
        # 然后通过线性层将其转换为词汇表大小的向量 (5, vocab_size)
        output = self.out(output.squeeze())
        # 返回输出和隐藏状态
        return output, hidden

    # 初始化隐藏状态的方法
    def init_hidden(self):
        # 返回一个全零的隐藏状态张量，维度为 (1, 1, 128)
        return torch.zeros(1, 1, 128).to(device)
# 构建训练函数
def train(epoch, train_log):
    # 构建词典，返回索引到词，词到索引，词汇数量，和语料索引列表
    index_to_word, word_to_index, word_count, corpus_idx = build_vocab()
    # 创建歌词数据集实例，每个输入序列的长度为 32
    lyrics = LyricsDataset(corpus_idx, 32)
    # 初始化文本生成模型，词汇表大小为 word_count
    model = TextGenerator(word_count).to(device)
    # TODO 正式进入这一阶段代码
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义 Adam 优化器，学习率为 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 打开日志文件用于写入
    file = open(train_log, 'w')
    # 开始训练循环
    for epoch_idx in range(epoch):
        # 数据加载器，打乱顺序，每次取 1 个样本
        lyrics_dataloader = DataLoader(lyrics, shuffle=True, batch_size=1)
        # 记录训练开始时间
        start = time.time()
        # 重置迭代次数
        iter_num = 0
        # 重置训练损失
        total_loss = 0.0
        # 遍历数据加载器中的数据
        for x, y in lyrics_dataloader:
            # 初始化隐藏状态
            hidden = model.init_hidden()
            # 前向传播计算输出和隐藏状态
            x, y = x.to(device), y.to(device)
            output, hidden = model(x, hidden)
            # 计算损失，y.squeeze() 去掉维度大小为1的维度
            # TODO 计算模型输出与实际目标之间的损失（误差）
            loss = criterion(output, y.squeeze())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            # 迭代次数加1
            iter_num += 1
            # 累加损失
            total_loss += loss.item()
        # 构建本次迭代的日志信息
        message = 'epoch %3s loss: %.5f time %.2f' % \
                  (epoch_idx + 1,  # 当前训练轮数
                   total_loss / iter_num,  # 平均损失
                   time.time() - start)  # 本轮训练时间
        # 打印日志信息
        print(message)
        # 写入日志文件
        file.write(message + '\n')
    # 关闭日志文件
    file.close()
    # 保存模型参数到文件
    torch.save(model.state_dict(), 'model/lyrics_model_%d.bin' % epoch)


# 构建预测函数
def predict(start_word, sentence_length, model_path):
    # 构建词典，返回索引到词，词到索引，词汇数量
    index_to_word, word_to_index, word_count, _ = build_vocab()
    # 构建文本生成模型实例，词汇表大小为 word_count
    model = TextGenerator(vocab_size=word_count).to(device)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(model_path))
    # 初始化隐藏状态
    hidden = model.init_hidden()
    try:
        # 将起始词转换为词索引
        word_idx = word_to_index[start_word]
    except:
        print("该词不在词典中, 请重新输入")
        return
    # 用于存储生成的句子（词索引序列）
    generate_sentence = [word_idx]
    # 生成长度为 sentence_length 的句子
    for _ in range(sentence_length):
        # 前向传播，获取模型输出和隐藏状态
        output, hidden = model(torch.tensor([[word_idx]]).to(device), hidden)
        # print("output: ", output)

        # 获取输出中概率最大的词的索引
        word_idx = int(torch.argmax(output).item())
        # 将该词索引添加到生成的句子中
        generate_sentence.append(word_idx)
    # 将生成的词索引序列转换为实际词并打印
    for idx in generate_sentence:
        print(index_to_word[idx], end='')
    print()


if __name__ == '__main__':
    train(10, train_log='log/train_log.txt')
    predict('可爱', 50, model_path='model/lyrics_model_1.bin')