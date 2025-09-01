import torch
import torch.nn as nn
import pickle
import torch.optim as optim

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self,inputs):
        inputs = self.linear1(inputs)
        outputs = self.linear2(inputs)
        return outputs
    

# 1. 保存模型参数
# 
# 模型参数保存为.pt文件

def test01 () :
    # 初始化模型参数
    model = Model(input_size=2,output_size=1)
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 定义要存储的模型参数
    state_dict = {
        'acc_score': 0.9,
        'loss_score': 0.1,
        'avg_score': 0.5,
        'inter_num': 100,
        'output_size': 1,
        'input_size': 2,
        'epoch': 100,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(state_dict, 'model/model-dict.pth')
    print(model.state_dict())


# 2. 加载模型参数
def test02 () :
    # 从磁盘加载模型参数加载到内存中
    model_params = torch.load('model/model-dict.pth')
    # 使用参数创建模型
    model = Model(input_size=model_params['input_size'],output_size=model_params['output_size'])  
    model.load_state_dict(model_params['model'])
    # 使用参数创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    optimizer.load_state_dict(model_params['optimizer'])
    ## 打印模型参数
    print('迭代次数：', model_params['epoch']) 
    print('准确率：', model_params['acc_score'])
    print('损失率：', model_params['loss_score'])
    print('平均值：', model_params['avg_score'])
    print('输入维度：', model_params['input_size'])
     

if __name__ == '__main__':
    test02()