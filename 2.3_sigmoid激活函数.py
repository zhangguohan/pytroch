import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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
            try:
                return fm.FontProperties(fname=font_path)
            except Exception as e:
                print(f"字体加载失败: {e}")
                continue
    
    # 如果没找到字体文件，尝试通过字体名称查找
    font_names = [
        'Noto Sans CJK SC', 'Noto Sans CJK', 'SimHei', 'Microsoft YaHei', 
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UKai CN',
        'AR PL UMing CN', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC',
        'Hiragino Sans GB'
    ]
    
    # 获取系统所有可用字体
    try:
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
    except Exception as e:
        print(f"获取系统字体列表失败: {e}")
        available_fonts = set()
    
    # 查找可用的中文字体
    for font_name in font_names:
        if font_name in available_fonts:
            print(f"使用系统字体: {font_name}")
            # 设置全局字体
            try:
                plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
                plt.rcParams['axes.unicode_minus'] = False
                return None  # 返回 None 表示使用全局设置
            except Exception as e:
                print(f"设置全局字体失败: {e}")
                continue
    
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


def plot_sigmoid_function():
    
    """绘制sigmoid函数图像"""
    
    _, axes = plt.subplots(1, 2)  
        # 绘制sigmoid函数
    x = torch.linspace(-20, 20, 100)
    y = torch.sigmoid(x)
        
    # 设置标题字体
       
    axes[0].set_title('sigmoid_函数图象', fontproperties=my_font)

    axes[0].plot(x, y) 
    axes[0].grid()   

    # 绘制导数图像
    x = torch.linspace(-20, 20, 100,requires_grad=True)
    y = torch.sigmoid(x).sum().backward()
        
    axes[1].set_title('sigmoid_导数图象', fontproperties=my_font)
    
    apply_chinese_font(axes[0].title, fontsize=14)
    axes[1].plot(x.detach(), x.grad)
    
    axes[1].grid()

        # 保存图像文件名
    image_filename = '1.10.2_sigmoid.png'
     
        # 保存图像（不使用fontproperties参数）
    try:
            plt.savefig(image_filename)
    except Exception as e:
            print(f"保存图像失败: {e}")
            # 尝试不使用字体属性保存
            plt.savefig(image_filename)
        
       




if __name__ == '__main__':
    plot_sigmoid_function()