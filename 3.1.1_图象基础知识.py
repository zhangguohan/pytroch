import numpy as np
import matplotlib.pyplot as plt


# 1.像素点的理解
def test():
    # img = np.zeros([200,200])
    # plt.imshow(img,cmap='gray',vmin=0,vmax=255)

    # plt.savefig('data/1.png')

    img1 = np.full([255,255],255)
    plt.imshow(img1,cmap='gray',vmin=0,vmax=255)
    plt.savefig('data/2.png')
    plt.show()

# 2.图象通道测试
def test2():
    import os
    # 从磁键的图片中读取图片
    img_path = 'data/钢铁侠.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片文件 {img_path} 不存在")
    
    try:
        img = plt.imread(img_path)
        print(img.shape)
        img = np.transpose(img, (2, 0, 1))
        print(img.shape)
        
        # 创建一个可写的副本
        img = np.copy(img)
        
        for i, channel in enumerate(img):
            plt.figure()
            plt.imshow(channel, cmap='gray')
            plt.savefig(f'data/channel_{i}.png')
            plt.close()
            
        # 透明通道
        plt.figure()
        plt.imshow(img[2], cmap='gray')
        # 修改显示透明通道
        img[2] = 100
        plt.imshow(img[2])
        plt.close()
        
        # 保存图片
        img = np.transpose(img, (1, 2, 0))
        plt.imsave('data/钢铁侠_alpha.png', img)
        
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        raise

if __name__ == '__main__':
    test2()