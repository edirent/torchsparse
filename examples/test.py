import time
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
import numpy as np

# 设置设备（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟 UNet 瓶颈层：输入张量形状 [B, C, H, W] = [2, 1024, 32, 32]
B, C_in, H, W = 2, 1024, 32, 32

# 生成全零输入，并构造一个80%空、20%有效的 mask
dense_input = torch.zeros(B, C_in, H, W, device=device)
mask = (torch.rand(B, H, W, device=device) < 0.2)  # True 表示有效数据

# 对 mask 为 True 的位置填入随机值
for b in range(B):
    # 填入随机值，保证每个有效位置在所有通道上都有数据
    dense_input[b, :, mask[b]] = torch.rand(C_in, mask[b].sum(), device=device)

# 定义 dense conv2d（标准卷积），输入通道1024，输出通道512，kernel_size=3, padding=1
dense_conv = nn.Conv2d(C_in, 512, kernel_size=3, padding=1).to(device)
# 预运行一次，确保权重初始化
_ = dense_conv(dense_input)

# 构造 sparse 表示：
# 对每个 batch，提取 mask 为 True 的位置，并构造4D 坐标 [batch, z, y, x]，其中 z 坐标全部置 0
coords_list = []
feats_list = []
for b in range(B):
    # torch.nonzero 返回 LongTensor，这里转换为 int 类型
    inds = torch.nonzero(mask[b], as_tuple=False).int()  # (n_points, 2)，顺序为 (row, col)
    if inds.numel() == 0:
        continue
    batch_idx = torch.full((inds.size(0), 1), b, dtype=torch.int, device=device)
    z_coord = torch.zeros((inds.size(0), 1), dtype=torch.int, device=device)  # z 坐标全部为 0
    # 调整顺序为 [batch, z, row, col]
    coords_sample = torch.cat([batch_idx, z_coord, inds], dim=1)  # (n_points, 4)
    coords_list.append(coords_sample)
    # 提取对应位置的特征：dense_input[b] 形状为 [C_in, H, W]，
    # 选取 mask[b] 对应位置后转置为 (n_points, C_in)
    feats_sample = dense_input[b, :, mask[b]].transpose(0, 1).contiguous()  
    feats_list.append(feats_sample)

if len(coords_list) == 0:
    raise ValueError("所有样本均为空！")

coords = torch.cat(coords_list, dim=0)  # (total_points, 4)
feats = torch.cat(feats_list, dim=0)      # (total_points, C_in)

# 构造 torchsparse 的 SparseTensor
sparse_input = torchsparse.SparseTensor(coords=coords, feats=feats).to(device)

# 定义 sparse conv：利用 spnn.Conv3d 模拟 2D 卷积，虚拟深度维度保持为1
sparse_conv = spnn.Conv3d(C_in, 512, kernel_size=3, stride=1, padding=1).to(device)
_ = sparse_conv(sparse_input)  # 预运行一次

# ------------------------------
# 打印输出
# ------------------------------
# Dense conv 输出
dense_out = dense_conv(dense_input)
print("Dense conv output shape:", dense_out.shape)
print("Dense conv output (sample):")
print(dense_out[0, :, :4, :4])  # 打印第一个 batch 的部分数据

# Sparse conv 输出
sparse_out = sparse_conv(sparse_input)
print("Sparse conv output feats shape:", sparse_out.feats.shape)
print("Sparse conv output coords shape:", sparse_out.coords.shape)
print("Sparse conv output feats (sample):")
print(sparse_out.feats[:5])  # 打印前 5 个有效数据

# ------------------------------
# 定义计时函数：dense conv
# ------------------------------
def time_dense_conv(iterations=100):
    # 预热
    for _ in range(10):
        _ = dense_conv(dense_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = dense_conv(dense_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iterations

# ------------------------------
# 定义计时函数：sparse conv
# ------------------------------
def time_sparse_conv(iterations=100):
    # 预热
    for _ in range(10):
        _ = sparse_conv(sparse_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = sparse_conv(sparse_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    end = time.time()
    return (end - start) / iterations

dense_time = time_dense_conv()
sparse_time = time_sparse_conv()

print("Dense conv (2D) average time per iteration: {:.6f} sec".format(dense_time))
print("Sparse conv (torchsparse 2D simulated) average time per iteration: {:.6f} sec".format(sparse_time))
