import torch
## print is gupy available?
print(torch.cuda.is_available())  # should return True if GPU is available

## print GPU name
## print(torch.cuda.get_device_name(0))

# data = torch.tensor([[1, 2, 3, 4, 5],[ 6, 7, 8, 9, 10]])
# data = data.to(torch.int32)
# # print(data)
# # print(data.dtype)
# print(data[0][4].item())

# data_a = torch.ones(3,3)
# data_b = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

# print(data_a + data_b)

# print(torch.sum(data_b,axis=1))



predict = torch.tensor([1,0,0,1,1,0,2,0,0,3,0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0])

lable = torch.tensor([1,0,0,1,1,0,2,0,0,3,0,2,3,0,1,0,0,1,1,0,1,0,1,2,0,0])

print(torch.mean((predict == lable).to(torch.float32)).item())
