import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.softmax(x, dim=0)
    print(y)