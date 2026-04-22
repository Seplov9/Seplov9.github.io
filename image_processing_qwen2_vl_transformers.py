# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Qwen2-VL."""

import math
from collections.abc import Iterable

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageInput,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


class Qwen2VLImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    min_pixels (`int`, *optional*, defaults to `56 * 56`):
        The min pixels of the image to resize the image.
    max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
        The max pixels of the image to resize the image.
    patch_size (`int`, *optional*, defaults to 14):
        The spatial patch size of the vision encoder.
    temporal_patch_size (`int`, *optional*, defaults to 2):
        The temporal patch size of the vision encoder.
    merge_size (`int`, *optional*, defaults to 2):
        The merge size of the vision encoder to llm encoder.
    """

    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int


def smart_resize(
    height: int,            # 原始图像高度 (H)
    width: int,             # 原始图像宽度 (W)
    factor: int = 28,       # 维度对齐因子（例如：Patch 大小），缩放后宽高需能被此数整除
    min_pixels: int = 56 * 56,               # 允许的最小像素总数 (H' * W')
    max_pixels: int = 14 * 14 * 4 * 1280     # 允许的最大像素总数 (H' * W')
):
    """
    对图像尺寸进行智能缩放：
    1. 宽高均需能被 factor 整除。
    2. 像素总数限制在 [min_pixels, max_pixels] 之间。
    3. 尽可能保持原始长宽比 (H/W)。
    """

    # --- 步骤 1: 极端长宽比校验 ---
    # 如果长宽比超过 200:1，通常认为图像异常或不适合模型处理，直接抛出异常
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    # --- 步骤 2: 初步四舍五入对齐 ---
    # 将原始宽高分别缩放到最接近的 factor 倍数
    # h_bar 维度: factor 的整数倍
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # --- 步骤 3: 像素总量上限控制 ---
    # 如果初步对齐后的像素总量超过了 max_pixels
    if h_bar * w_bar > max_pixels:
        # 计算缩放比例 beta：基于面积比的开方，确保缩放后面积近似等于 max_pixels
        # beta = sqrt(当前总像素 / 目标最大像素)
        beta = math.sqrt((height * width) / max_pixels)
        
        # 使用 beta 缩小尺寸：
        # 1. height / beta 得到等比例缩小的理论高度
        # 2. floor(... / factor) * factor 确保向下取整到 factor 的倍数
        # 3. max(factor, ...) 确保维度至少为 factor，防止变成 0
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)

    # --- 步骤 4: 像素总量下限控制 ---
    # 如果初步对齐后的像素总量小于 min_pixels
    elif h_bar * w_bar < min_pixels:
        # 计算扩大的比例 beta
        # beta = sqrt(目标最小像素 / 当前总像素)
        beta = math.sqrt(min_pixels / (height * width))
        
        # 使用 beta 放大尺寸：
        # 1. height * beta 得到等比例放大的理论高度
        # 2. ceil(... / factor) * factor 确保向上取整到 factor 的倍数
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # 返回最终调整后的维度 (H', W')
    return h_bar, w_bar


@auto_docstring
class Qwen2VLImageProcessor(TorchvisionBackend):
    do_resize = True
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
    default_to_square = False
    do_rescale = True
    do_normalize = True
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    valid_kwargs = Qwen2VLImageProcessorKwargs
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Unpack[Qwen2VLImageProcessorKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        size = self.size if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: int | Iterable[int] | dict[str, int] | SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return kwargs

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[Qwen2VLImageProcessorKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],              # 输入图像列表，是一个list
        # images[0].shape: [3, 1365, 2048]
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int,                          # 空间补丁大小 (如 14)
        temporal_patch_size: int,                 # 时间补丁大小 (如 2，用于视频)
        merge_size: int,                          # 合并系数 (通常为 2，对应 2x2 的 patch 合并)
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        # 1. 将图像按分辨率分组，减少不同尺寸带来的 pad 开销
        '''
        grouped_images是一个字典dict，为kv对， 格式为：{ 图像形状（分辨率） ： 堆叠后的图片张量 }
        grouped_images_index也是一个字典dict，格式为：{ 原始图像序号 : (张量形状组, 组内位置索引) }
        原始索引 (Key)	分组后的定位 (Value)	含义
        0	            ((3, 224, 224), 0)	    原来第 0 张图，现在在 224x224 组的第 0 个位置
        1	            ((3, 512, 512), 0)	    原来第 1 张图，现在在 512x512 组的第 0 个位置
        2	            ((3, 224, 224), 1)	    原来第 2 张图，现在在 224x224 组的第 1 个位置
        '''
        
        '''
        image * 1
        grouped_images.keys()[0]: torch.Size([1365, 2048])] (pdb grouped_images.keys())
        grouped_images.values()[0].shape: [1, 3, 1365, 2048] (pdb [v.shape for v in grouped_images.values()])
        grouped_images_index: {0: (torch.Size([1365, 2048]), 0)}
        '''
        
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        # 遍历每个分辨率组进行缩放
        for shape, stacked_images in grouped_images.items():
            # stacked_images 维度: [B_group, (T,) C, H, W]
            height, width = stacked_images.shape[-2:]
            
            if do_resize:
                # 2. 计算满足补丁和合并倍数的最佳缩放尺寸
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,  # 确保能被 patch_size * merge_size 整除
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                # 执行缩放: [B_group, (T,) C, H, W] -> [B_group, (T,) C, resized_H, resized_W]
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_images_grouped[shape] = stacked_images
            '''
            image * 1
            resized_images_grouped.keys()[0].shape: torch.Size([1365, 2048])
            resized_images_grouped.values()[0].shape: [1, 3, 1376, 2048]
            '''
        
        # 3. 将缩放后的图像还原回原始输入的顺序列表
        # reorder_images作用是：根据文本中出现的图像占位符顺序，重新排列图像数据的存储顺序，返回list
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        # image * 1: resized_images[0].shape: [3, 1376, 2048]

        # 4. 再次分组（针对缩放后的尺寸），准备进行像素值处理
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        '''
        image * 1
        grouped_images.keys()[0]: torch.Size([1376, 2048])] (pdb grouped_images.keys())
        grouped_images.values()[0].shape: [1, 3, 1376, 2048] (pdb [v.shape for v in grouped_images.values()])
        grouped_images_index: {0: (torch.Size([1376, 2048]), 0)}
        '''
        
        processed_images_grouped = {}
        processed_grids = {}

        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # 归一化与缩放 (如 /255.0): 维度维持 [B_group, (T,) C, H, W]
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            # image * 1: patches.shape: [1, 3, 1376, 2048]
            
            # 如果是图像 (ndim=4, [B, C, H, W])，增加时间轴 T=1 -> [B, 1, C, H, W]
            if patches.ndim == 4:
                patches = patches.unsqueeze(1)
            # image * 1: patches.shape: [1, 1, 3, 1376, 2048]
            
            # 5. 视频时间轴对齐：如果 T 不是 temporal_patch_size 的倍数，则对最后一帧进行填充(Padding)
            if patches.shape[1] % temporal_patch_size != 0:
                # 选取最后一帧并重复: [B, 1, C, H, W]
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                # image * 1: repeats.shape: [1, 1, 3, 1376, 2048]
                
                # 拼接后 T 变为 temporal_patch_size 的倍数
                patches = torch.cat([patches, repeats], dim=1)
                # image * 1: patches.shape: [1, 2, 3, 1376, 2048]
            
            batch_size, grid_t_raw, channel = patches.shape[:3] # grid_t_raw 是 T 维度
            grid_t = grid_t_raw // temporal_patch_size        # 时间维度的 patch 数量
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size # 空间维度的 patch 数量
            # image * 1: (grid_t, grid_h, grid_w) = (1, 86, 128)

            # 6. 核心重塑 (Reshape): 将图像切割成 3D Patches
            # 维度从 [B, T, C, H, W] 拆解为极其细分的维度以备后续 Permute
            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # image * 1: patches.shape: [1, 1, 2, 3, 43, 2, 16, 64, 2, 16]
            
            # 7. 维度重排 (Permute): 重新排列维度以将 patch 内部的数据聚合
            # 目标是把时间块、空间块的像素拉到最后，以便 Flatten
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            # image * 1: patches.shape: [1, 1, 43, 64, 2, 2, 3, 2, 16, 16]
            
            # 8. 展平 (Flatten):
            # 最终维度: [B, (grid_t * grid_h * grid_w), (C * T_patch * H_patch * W_patch)]
            flatten_patches = patches.reshape(
                batch_size,  # (1)
                grid_t * grid_h * grid_w,  # (1, 43, 64, 2, 2)
                channel * temporal_patch_size * patch_size * patch_size,  # (3, 2, 16, 16)
            )
            '''
            image * 1: 
            batch_size: (1)
            grid_t * grid_h * grid_w: (1, 43, 64, 2, 2)
            channel * temporal_patch_size * patch_size * patch_size: (3, 2, 16, 16)
            flatten_patches.shape: [1, 11008, 1536]
            '''

            processed_images_grouped[shape] = flatten_patches
            # 记录当前分辨率下的补丁网格结构 [T, H, W]
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        # 9. 还原顺序并合并 Batch
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids_ordered = reorder_images(processed_grids, grouped_images_index)

        '''
        image *1
        processed_images[0].shape: [11008, 1536]
        processed_grids_ordered[0]: [1, 86, 128]
        '''
        
        # pixel_values 维度: [Total_B, Total_Patches, Patch_Dim]
        pixel_values = torch.cat(processed_images, dim=0)
        # image_grid_thw 维度: [Total_B, 3] (记录每张图的 T, H, W 网格数)
        image_grid_thw = torch.tensor(processed_grids_ordered, dtype=torch.long)

        # image * 1: pixel_values.shape: [11008, 1536], image_grid_thw.shape: [1, 3]

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, 
            tensor_type=return_tensors
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Note: Do not remove this method! It is used by vLLM to infer the number of patches and placeholders
        without an image input.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        min_pixels = images_kwargs["min_pixels"] if "min_pixels" in images_kwargs else self.size["shortest_edge"]
        max_pixels = images_kwargs["max_pixels"] if "max_pixels" in images_kwargs else self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Qwen2VLImageProcessor"]
