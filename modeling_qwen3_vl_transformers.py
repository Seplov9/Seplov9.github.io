import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, torch_compilable_check
from ...utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig


@dataclass  # 指示 Python 将该类自动转换为数据类，自动生成 __init__、__repr__ 等方法
@auto_docstring  # 这是一个装饰器（常见于某些框架），用于根据类属性自动生成或补充文档字符串
class BaseModelOutputWithDeepstackFeatures(BaseModelOutputWithPooling):
    r"""
    BaseModelOutputWithDeepstackFeatures 类继承自 BaseModelOutputWithPooling。
    它不仅包含基础模型输出（如 hidden_states）和池化输出（pooled_output），
    还额外增加了对 Deepstack（深度堆叠）层特征的存储支持。

    参数说明：
    deepstack_features (`List[torch.FloatTensor]`, *可选*):
        来自模型“深度堆叠”层的隐藏状态（特征图）列表。
    """

    # 定义属性 deepstack_features：
    # 类型为 torch.FloatTensor 的列表，或者为 None
    # 默认值设置为 None
    deepstack_features: list[torch.FloatTensor] | None = None


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 模型的隐藏层维度（输入/输出维度），例如 1024
        self.hidden_size = config.hidden_size 
        
        # 中间层扩充维度，通常是 hidden_size 的 4 倍（如 4096）
        self.intermediate_size = config.intermediate_size 
        
        # 第一层全连接：从 hidden_size 映射到 intermediate_size
        # 权重维度: [intermediate_size, hidden_size]
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        
        # 第二层全连接：将维度还原回 hidden_size
        # 权重维度: [hidden_size, intermediate_size]
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        
        # 激活函数，通常使用 GeLU 或 SwiGLU 变体
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        # 输入 hidden_state 维度: [Batch, Sequence_Length, hidden_size]
        
        # 1. 第一层映射：[B, S, hidden_size] -> [B, S, intermediate_size]
        x = self.linear_fc1(hidden_state)
        
        # 2. 激活函数处理：维度保持 [B, S, intermediate_size]
        x = self.act_fn(x)
        
        # 3. 第二层映射：[B, S, intermediate_size] -> [B, S, hidden_size]
        x = self.linear_fc2(x)
        
        return x


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # 提取配置参数
        self.patch_size = config.patch_size  # 空间维度的 Patch 大小 (如 14)
        self.temporal_patch_size = config.temporal_patch_size  # 时间维度的 Patch 大小 (如 2)
        self.in_channels = config.in_channels  # 输入通道数 (如 3)
        self.embed_dim = config.hidden_size  # 映射后的 Embedding 维度 (如 1024)

        # 卷积核大小设置为 [时间步, 高度, 宽度]
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        
        # 使用 3D 卷积进行线性投影。步长(stride)等于核大小(kernel_size)，确保 Patch 之间不重叠
        # 输入维度: (Batch, C_in, T, H, W)
        # 输出维度: (Batch, Embed_dim, T', H', W') 其中 T'=T/temporal_patch_size, H'=H/patch_size
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 获取权重的数据类型（如 float16 或 bfloat16），确保计算精度一致
        target_dtype = self.proj.weight.dtype
        
        # 【重塑维度】 将输入展平为 Conv3d 期待的 5D Tensor
        # 假设原始 hidden_states 为 (Total_Patches, -1)
        # 变换后维度: (N, C_in, T_patch, H_patch, W_patch) 
        # 此处的 N 是为了处理被切分后的局部块
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        
        # 【执行投影】 
        # 1. hidden_states.to(dtype=target_dtype): 类型转换
        # 2. self.proj(...): 执行 3D 卷积。由于输入刚好等于核大小，卷积后空间维度变为 1x1x1
        #    卷积输出维度: (N, Embed_dim, 1, 1, 1)
        # 3. .view(-1, self.embed_dim): 展平为二维矩阵
        #    最终输出维度: (N, Embed_dim)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    # 定义类型注解，告知 Linter inv_freq 是一个 Tensor 类型的缓冲区
    inv_freq: torch.Tensor  

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim      # 嵌入维度（通常为 head_dim 的一部分）
        self.theta = theta  # 基数（Base），用于控制不同维度的旋转频率

        # 计算逆频率向量 (Inverse Frequency)
        # torch.arange(0, dim, 2): 生成 [0, 2, 4, ..., dim-2]，维度为 (dim/2,)
        # 计算公式: 1.0 / (theta ** (i / dim))
        # inv_freq 维度: [dim / 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        
        # 将 inv_freq 注册为 buffer，不会被视为模型参数（不更新梯度）
        # persistent=False 表示该 buffer 不会保存到 state_dict 中
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        # 生成位置索引序列: [0, 1, 2, ..., seqlen-1]
        # seq 维度: [seqlen]
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        # 计算外积 (Outer Product)
        # seq (seqlen,) 与 inv_freq (dim/2,) 相乘
        # freqs 维度: [seqlen, dim / 2]
        freqs = torch.outer(seq, self.inv_freq)
        
        # 返回计算好的频率矩阵，后续通常会配合 sin/cos 使用
        # 返回维度: [seqlen, dim / 2]
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        # 计算合并后的隐藏层维度。
        # 维度变化：hidden_size = D * (S^2)，其中 S 是 spatial_merge_size（通常为 2）。
        # 如果 S=2，则维度扩大 4 倍。
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        
        # 是否在像素重组（Pixel Shuffle/Reshape）之后才进行归一化。
        self.use_postshuffle_norm = use_postshuffle_norm
        
        # 归一化层：
        # 如果 postshuffle，则 LayerNorm 作用于扩大后的维度 (D * S^2)；
        # 否则作用于原始维度 D。
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        
        # 第一层线性变换：输入和输出维度均为 (D * S^2)。
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 激活函数，通常为 GELU。
        self.act_fn = nn.GELU()
        
        # 第二层线性变换：将特征投影到 LLM 接收的维度 (config.out_hidden_size)。
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的初始维度: (B, L, D) 
        # 其中 L = H * W (原始图像切分后的宽高)
        
        # 第一步：LayerNorm 与 Reshape (Pixel Shuffle 逻辑)
        # 如果 use_postshuffle_norm 为 False:
        #   1. 先对 x (B, L, D) 进行 norm -> 维度保持 (B, L, D)
        #   2. 执行 view(-1, hidden_size) -> 维度变为 (B, L/S^2, D*S^2)
        # 如果 use_postshuffle_norm 为 True:
        #   1. 先执行 view(-1, hidden_size) -> 维度变为 (B, L/S^2, D*S^2)
        #   2. 对变换后的张量进行 norm -> 维度保持 (B, L/S^2, D*S^2)
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        
        # 此时 x 的维度: (B * L/S^2, D * S^2)
        
        # 第二步：通过 MLP 结构进行特征融合与维度投影
        # linear_fc1: (B * L/S^2, D * S^2) -> (B * L/S^2, D * S^2)
        # act_fn: 维度不变
        # linear_fc2: (B * L/S^2, D * S^2) -> (B * L/S^2, out_hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        
        # 返回结果维度: (B * L/S^2, out_hidden_size)
        # 注：通常在实际输出前会根据 Batch Size 重新 reshape 回 (B, L/S^2, out_hidden_size)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
