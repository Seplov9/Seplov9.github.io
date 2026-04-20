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


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # 记录原始数据类型（通常是 bfloat16 或 float16），以便最后还原
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype

    # 将 q 和 k 转换为 float32 以保证旋转运算的数值稳定性
    # q, k 维度: [B, L, H, D] (Batch, Length, Heads, Head_dim)
    q, k = q.float(), k.float()

    # 对 cos 和 sin 进行升维，增加一个维度以对齐 Head_dim 之前的维度
    # 输入 cos/sin 维度通常为: [L, D] 或 [B, L, D]
    # unsqueeze(-2) 后变为: [L, 1, D] 或 [B, L, 1, D]，确保能通过广播机制应用到所有 Head (H) 上
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()

    # 应用旋转变换公式：q_new = q * cos(mθ) + rotate_half(q) * sin(mθ)
    # rotate_half 的作用是将向量的后半部分取负并与前半部分交换，模拟复数乘法
    # q_embed 维度: [B, L, H, D]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    # 对 Key (k) 执行相同的旋转操作
    # k_embed 维度: [B, L, H, D]
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 将结果转换回原始的高性能低精度格式（如 bf16）
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    此函数的作用是将 KV 头的数量重复 n_rep 次，以匹配 Query 头的数量。
    输入维度: (batch, num_key_value_heads, seqlen, head_dim)
    n_rep: 重复次数，通常为 num_attention_heads // num_key_value_heads
    """
    
    # 1. 获取输入张量的原始维度信息
    # batch: 批量大小, num_key_value_heads: KV头的数量, slen: 序列长度, head_dim: 每个头的维度
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # 2. 如果重复次数为1，说明 KV 头数量已经等于 Query 头数量（即标准的 Multi-Head Attention）
    # 直接返回原张量，无需操作
    if n_rep == 1:
        return hidden_states
    
    # 3. 维度扩展 (Expansion)
    # hidden_states[:, :, None, :, :] 在第 2 维（索引从0开始）插入一个新维度。
    # 插入后维度变为: (batch, num_key_value_heads, 1, slen, head_dim)
    # .expand(...) 将这个新增的维度从 1 广播（重复）到 n_rep。
    # 此时维度变为: (batch, num_key_value_heads, n_rep, slen, head_dim)
    # 注：expand 不会分配新内存，只是创建了视图，效率极高。
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # 4. 维度重塑 (Reshape)
    # 将 num_key_value_heads 和 n_rep 合并到同一个维度。
    # 结果维度: (batch, num_key_value_heads * n_rep, slen, head_dim)
    # 这里的 num_key_value_heads * n_rep 实际上就等于总的 num_attention_heads。
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,              # 传入的注意力模块实例，用于获取配置参数（如 head 数量）
    query: torch.Tensor,           # 查询向量 [batch, num_heads, seq_len, head_dim]
    key: torch.Tensor,             # 键向量 [batch, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,           # 值向量 [batch, num_kv_heads, seq_len, head_dim]
    attention_mask: torch.Tensor | None, # 掩码，用于遮蔽无效位置（如 Padding 或 Causal 掩码）
    scaling: float,                # 缩放因子，通常为 1 / sqrt(head_dim)
    dropout: float = 0.0,          # Dropout 概率
    **kwargs: Unpack[TransformersKwargs],
):
    # 1. 针对 Grouped Query Attention (GQA)，将 Key 重复扩展以匹配 Query 的 Head 数量
    key_states = repeat_kv(key, module.num_key_value_groups)
    # 2. 同理，将 Value 重复扩展
    value_states = repeat_kv(value, module.num_key_value_groups)

    # 3. 核心计算：Q 乘以 K 的转置，计算注意力分数（矩阵乘法）
    # 结果维度: [batch, num_heads, seq_len, seq_len]
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # 4. 如果有掩码（如处理变长输入或因果遮蔽），将其加到分数上（通常掩码处为极负数）
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 5. 在最后一个维度做 Softmax，使权重归一化到 [0, 1] 且和为 1
    # 强制转为 float32 计算以保证数值稳定性，最后转回 Query 的原始精度
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # 6. 对注意力权重进行 Dropout 随机失活，防止过拟合
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # 7. 将权重作用于 Value 向量上，得到最终的上下文表示
    # 结果维度: [batch, num_heads, seq_len, head_dim]
    attn_output = torch.matmul(attn_weights, value_states)

    # 8. 调整维度顺序，将多头拼接到一起，并确保内存连续（以便后续 View/Reshape 操作）
    # 结果维度: [batch, seq_len, num_heads, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size      # 隐藏层维度，例如 1280
        self.num_heads = config.num_heads  # 注意力头数，例如 16
        self.head_dim = self.dim // self.num_heads # 每个头的维度，例如 1280/16 = 80
        
        # 为了兼容不同的 Attention 实现（如 GQA 逻辑），设置 KV 组数为 1
        self.num_key_value_groups = 1  
        
        # 定义 QKV 投影层：将输入映射到 3 倍维度 (Q, K, V)
        # [dim] -> [3 * dim]
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        
        # 定义输出投影层：将注意力结果映射回原始维度
        # [dim] -> [dim]
        self.proj = nn.Linear(self.dim, self.dim)
        
        # 缩放因子：1 / sqrt(head_dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False # 视觉编码器通常使用双向注意力，非因果
        
    def forward(
        self,
        hidden_states: torch.Tensor, # 输入特征: [L, dim]
        cu_seqlens: torch.Tensor,    # 累积序列长度: [batch_size + 1]，用于区分拼在一起的不同图片
        # 假设 Batch 中有 3 张图片，patch 数量分别为：图片 0: 5 个 token图片 1: 3 个 token图片 2: 4 个 token拼接后的总长度 L = 5 + 3 + 4 = 12。那么对应的 cu_seqlens 为：[0, 5, 8, 12]
        rotary_pos_emb: torch.Tensor | None = None, # 备用 RoPE
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, # (cos, sin) 
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0] # L
        
        # 1. 线性投影并切分为 Q, K, V
        # self.qkv(hidden_states) -> [L, 3 * dim]
        # .reshape(L, 3, num_heads, head_dim)
        # .permute(1, 0, 2, 3) -> [3, L, num_heads, head_dim]
        # .unbind(0) -> 得到三个 [L, num_heads, head_dim] 的张量
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        # 2. 应用视觉旋转位置编码 (Vision RoPE)
        # cos, sin 维度通常为 [L, head_dim]
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # 3. 维度转换，为后续 Attention 接口做准备
        # transpose(0, 1) -> [num_heads, L, head_dim]
        # unsqueeze(0) -> [1, num_heads, L, head_dim]
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        # 获取具体的注意力实现函数
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # --- 情况 1: 使用 Flash Attention ---
            # 通过 cu_seqlens 计算当前批次中最大的单序列长度
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            
            # Flash Attention 能够利用 cu_seqlens 直接在打包好的 [1, num_heads, L, head_dim] 上计算
            # 它内部会根据索引切分不同图片的注意力范围，防止跨图片干扰
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens, # 关键：指示序列边界
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # --- 情况 2: 常规实现 (Eager/SDPA(Scaled Dot Product Attention)) ---
            # 由于不支持 cu_seqlens，需要手动把 L 维按长度切开
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            
            # 将 Q, K, V 分别切成 list，每个元素对应一张图片
            # 输入维度：此时的 query_states 等张量的维度是 [1, num_heads, L, head_dim]。
            # 操作：在 dim=2（即 L 这一维）上，按照刚才算出的 lengths 长度进行切片。
            # 结果：将一个巨大的张量切分成一个张量列表（List of Tensors）。
            # q_list[0] 形状：[1, num_heads, 5, head_dim] （图片 0）
            # q_list[1] 形状：[1, num_heads, 3, head_dim] （图片 1）
            # q_list[2] 形状：[1, num_heads, 4, head_dim] （图片 2）
            # q 的 shape list: [[1, num_heads, len_i, head_dim], ...]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            # 循环计算每一张图片的注意力
            attn_outputs = [
                attention_interface(
                    self, q, k, v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            # 拼接结果: [1, L, num_heads, head_dim]
            attn_output = torch.cat(attn_outputs, dim=1)

        # 4. 恢复维度并投影
        # attn_output 原始维度: [1, L, num_heads, head_dim] 或 Flash 输出的变体
        # reshape(seq_length, -1) -> [L, dim]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        
        # 最后的线性映射 [L, dim] -> [L, dim]
        attn_output = self.proj(attn_output)
        
        return attn_output


# 继承自 GradientCheckpointingLayer，旨在支持梯度检查点以节省显存
class Qwen3VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        # 初始化第一层归一化：用于 Self-Attention 之前
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # 初始化第二层归一化：用于 MLP 之前
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # 视觉自注意力模块：负责 Patch 之间的空间信息交互
        self.attn = Qwen3VLVisionAttention(config=config)
        
        # 多层感知机：负责对每个 Patch 的特征进行非线性映射与维度变换
        self.mlp = Qwen3VLVisionMLP(config=config)

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor, # 输入维度: [N, D]
        cu_seqlens: torch.Tensor,    # 维度: [Batch_Size + 1], 记录每张图片在 N 中的起始位置
        rotary_pos_emb: torch.Tensor | None = None, # 维度: [N, Head_Dim], 旋转位置编码
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None, # 可选的 2D/3D 位置偏移
        **kwargs,
    ) -> torch.Tensor:
        r"""
        cu_seqlens: 用于处理非填充（unpadded）的变长数据，提高计算效率。
        rotary_pos_emb: 应用于 Q 和 K 的旋转位置编码，增强模型对空间位置的感知。
        """
        
        # --- 第一阶段：注意力机制与残差连接 ---
        # 1. self.norm1(hidden_states): [N, D] -> [N, D] (层归一化，稳定梯度)
        # 2. self.attn(...): 计算注意力，输出维度 [N, D]
        # 3. hidden_states + ...: 残差连接，防止梯度消失
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # --- 第二阶段：前馈网络（MLP）与残差连接 ---
        # 1. self.norm2(hidden_states): [N, D] -> [N, D]
        # 2. self.mlp(...): [N, D] -> [N, D] (通常内部会先升维再降维)
        # 3. hidden_states + ...: 再次残差连接
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

        # 返回更新后的特征表示，维度: [N, D]
        return hidden_states
