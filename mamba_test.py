import torch
from mamba_ssm import Mamba

# 设置批量大小、序列长度和维度
batch, length, dim = 2, 64, 16

# 创建一个随机输入张量x
x = torch.randn(batch, length, dim).to("cuda")

# 初始化Mamba模型
model = Mamba(
    d_model=dim,    # 模型维度 d_model
    d_state=16,     # SSM状态扩展系数
    d_conv=4,       # 本地卷积宽度
    expand=2,       # 块扩展系数
).to("cuda")

# 打印模型参数概要
print("Model Summary:")
print(model)

# 获取模型的输出y
y = model(x)

# 确认输出形状是否与输入相同
assert y.shape == x.shape

# 打印输出形状来确认模型运行无误
print("Output shape:", y.shape)
print("Mamba Model is running successfully.")
