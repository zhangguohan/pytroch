import torch
import torch.nn as nn
import pickle


class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self,inputs):
        inputs = self.linear1(inputs)
        outputs = self.linear2(inputs)
        return outputs
    
## 保存模型
def test01 ():
    model = Model(input_size=2,output_size=1)
    torch.save(model, 'model/model.pkl',pickle_module=pickle,pickle_protocol=2)


## 加载模型
def test02 ():
    model = torch.load('model/model.pkl',pickle_module=pickle,map_location='cpu')
    print(model)

if __name__ == '__main__':
    #test01()
    test02()  