import torch
import torch.nn as nn
import jieba

if __name__ == '__main__':
    text = '今天天气不错,底气、从容；信心、优势！财政政策发力空间依然充足'
    # 1.分词器
    words = jieba.lcut(text)
    print(words)

    # 2.构建词表
    word_to_ix = {} # 给定一个单词，返回单词的索引
    ix_to_word = {} # 给定一个索引，返回单词
    # 去重掉重复的
    unique_words = list(set(words)) 
    for i, word in enumerate(unique_words):
        word_to_ix[word] = i
        ix_to_word[i] = word
    # 3. 构建词向量
    embedding = nn.Embedding(num_embeddings=len(ix_to_word), embedding_dim=4)
   
   # 4. 获取词向量
   # 获取索引\及索引对应的单词
    for i, word in enumerate(words):
        ix = word_to_ix[word]
        print(ix, ix_to_word[ix])
        print(embedding(torch.tensor([ix])))
