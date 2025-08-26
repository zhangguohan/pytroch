def test02():

# 创建一个形状为 (2, 3, 4) 的随机张量
    torch.manual_seed(0)
    data = torch.randint(0,10,[2, 3])

    print("原始张量:")
    print(data)
    print(data.shape)  # 输出: torch.Size([2, 3])

    # 使用 permute 函数将维度顺序改为 (1, 0)
    permuted_tensor = data.permute(1, 0) 

    print("\n排列后的张量:")
    print(permuted_tensor)
    print(permuted_tensor.shape)  # 输出: torch.Size([3, 2])