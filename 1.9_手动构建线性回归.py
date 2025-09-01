import random
import torch
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# 自动查找并设置中文字体支持
def setup_chinese_font():
    """
    自动查找系统中可用的中文字体并设置
    """
    import os
    import platform
    
    # 定义可能的中文字体路径和名称
    font_candidates = []
    system = platform.system()
    
    if system == "Linux":
        # Linux 系统常见中文字体路径
        linux_fonts = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
        font_candidates.extend(linux_fonts)
    elif system == "Windows":
        # Windows 系统中文字体
        windows_fonts = [
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc", 
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
        ]
        font_candidates.extend(windows_fonts)
    elif system == "Darwin":  # macOS
        # macOS 系统中文字体
        macos_fonts = [
            "/Library/Fonts/Arial Unicode MS.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
        font_candidates.extend(macos_fonts)
    
    # 查找第一个存在的字体文件
    for font_path in font_candidates:
        if os.path.exists(font_path):
            print(f"找到中文字体: {font_path}")
            return fm.FontProperties(fname=font_path)
    
    # 如果没找到字体文件，尝试通过字体名称查找
    font_names = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'Microsoft YaHei', 
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UKai CN',
        'AR PL UMing CN', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC',
        'Hiragino Sans GB'
    ]
    
    # 获取系统所有可用字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 查找可用的中文字体
    for font_name in font_names:
        if font_name in available_fonts:
            print(f"使用系统字体: {font_name}")
            # 设置全局字体
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return None  # 返回 None 表示使用全局设置
    
    # 如果都没找到，使用默认字体并给出警告
    print("警告: 未找到合适的中文字体，中文可能显示为方块")
    print("建议安装中文字体包，如:")
    if system == "Linux":
        print("  sudo apt-get install fonts-noto-cjk")
    elif system == "Windows":
        print("  系统应该已包含中文字体")
    elif system == "Darwin":
        print("  系统应该已包含中文字体")
    
    return None

# 自动设置中文字体
my_font = setup_chinese_font()

# 如果找到了字体文件，创建字体属性函数
def apply_chinese_font(text_element, fontsize=12):
    """应用中文字体到文本元素"""
    if my_font is not None:
        return {'fontproperties': my_font, 'fontsize': fontsize}
    else:
        return {'fontsize': fontsize}

# 创建数据集
def create_dataset(n_samples=1000,
                    n_features=1,
                    noise=10,
                    coef=True,
                    random_state=0,
                    bias=14.5):
    x, y, coef= make_regression(n_samples=n_samples, coef=coef, n_features=n_features, noise=noise)
    # 将构建的数据转成张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y, coef


# 构建数据加载器
def create_dataloader(x, y, batch_size):
    # 计算下样本数
    data_len = len(x)
    # 构建数据索引
    data_index = list(range(data_len))
    # 数据集打乱
    random.shuffle(data_index)
    #计算总的batch数
    total_batch = data_len // batch_size
    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size
        # 获取当前batch的数据
        batch_x = x[start:end]
        batch_y = y[start:end]
        yield batch_x, batch_y

# 假设函数
w = torch.tensor(0.1, requires_grad=True,dtype=torch.float64)
b = torch.tensor(0.1, requires_grad=True,dtype=torch.float64)
def linear_regression(x):
    return w * x + b

# 损失函数
def square_loss(y_pred, y):
    return torch.mean((y_pred - y) ** 2)


# 优化方法
def sgd(lr=1e-2):
# 此处除以batch_size，使用的是批样本的平均梯度值 
    batch_size = 16
    if w.grad is not None and b.grad is not None:
        w.data = w.data - lr * w.grad.data / batch_size
        b.data = b.data - lr * b.grad.data / batch_size
        # 清零梯度
        w.grad.data.zero_()
        b.grad.data.zero_()

def test():
    x, y,coef = create_dataset()
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y)
    plt.title('数据集散点图',  fontdict=apply_chinese_font(None, 16))
    plt.xlabel('特征值 (X)',  fontdict=apply_chinese_font(None, 16))
    plt.ylabel('目标值 (Y)',  fontdict=apply_chinese_font(None, 16))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('data.png', dpi=300, bbox_inches='tight')
    plt.show()

# 训练
def train():
    # 加载数据
    x, y, coef = create_dataset()
    # 定义训练参数
    epochs = 100
    learning_rate = 1e-2
    # 存储训练信息
    epoch_loss = []
    
    print("开始训练...")
    for epoch in range(epochs):
        total_loss = 0.0
        train_samples = 0
        
        for train_x, train_y in create_dataloader(x, y, batch_size=16):
            #1. 将训练数据传入模型中
            y_pred = linear_regression(train_x)
            #2. 计算预测值和真实值的的平方误差
            loss = square_loss(y_pred, train_y.reshape(-1, 1))
            total_loss += loss.item() * len(train_y)  # 乘以批次大小以正确计算加权平均
            train_samples += len(train_y)
            # 3. 梯度清零
            if w.grad is not None: 
                w.grad.data.zero_()
            if b.grad is not None:
                b.grad.data.zero_()
            # 4. 反向传播
            loss.backward()

            # 5. 参数更新
            sgd(learning_rate)

        # 6. 记录每个epoch的平均损失
        avg_loss = total_loss / train_samples
        epoch_loss.append(avg_loss)
        if (epoch + 1) % 10 == 0:  # 每10轮打印一次
            print('轮次 %d, 损失: %.10f' % (epoch+1, avg_loss))
    
    print(f"训练完成! 最终参数: w={w.item():.4f}, b={b.item():.4f}")
    # 修复 NumPy 2.0 兼容性问题：确保 coef 是标量
    coef_scalar = float(coef) if hasattr(coef, 'item') else coef
    print(f"真实参数: w={coef_scalar:.4f}, b=14.5")
    
    # 绘制训练结果图
    plt.figure(figsize=(12, 5))
    
    # 子图1: 数据散点图和回归直线
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.6, s=20)
    
    # 绘制预测值的直线
    plot_x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * w + b for v in plot_x])
    # 修复 NumPy 2.0 兼容性问题：确保 coef 是标量
    coef_scalar = float(coef) if hasattr(coef, 'item') else coef
    y2 = torch.tensor([v * coef_scalar + 14.5 for v in plot_x])
    
    plt.plot(plot_x, y1, color='red', linewidth=2, label='训练结果')
    plt.plot(plot_x, y2, color='green', linewidth=2, label='真实函数')
    
    plt.title('线性回归结果对比',  fontdict=apply_chinese_font(None, 16))
    plt.xlabel('特征值 (X)',  fontdict=apply_chinese_font(None, 16))
    plt.ylabel('目标值 (Y)',  fontdict=apply_chinese_font(None, 16))
    plt.grid(True, alpha=0.3)
    
    # 设置图例字体
    if my_font is not None:
        plt.legend(prop=my_font, fontsize=10)
    else:
        plt.legend(fontsize=10)
    
    # 子图2: 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), epoch_loss, 'b-', linewidth=2)
    plt.title('训练损失曲线',  fontdict=apply_chinese_font(None, 16))
    plt.xlabel('训练轮次 (Epochs)',  fontdict=apply_chinese_font(None, 16))
    plt.ylabel('均方误差损失',  fontdict=apply_chinese_font(None, 16))
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('result.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 单独绘制损失曲线（保持原有功能）
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), epoch_loss, 'b-', linewidth=2)
    plt.title('训练损失曲线',  fontdict=apply_chinese_font(None, 16))
    plt.xlabel('训练轮次 (Epochs)',  fontdict=apply_chinese_font(None, 16))
    plt.ylabel('均方误差损失',  fontdict=apply_chinese_font(None, 16))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    train()