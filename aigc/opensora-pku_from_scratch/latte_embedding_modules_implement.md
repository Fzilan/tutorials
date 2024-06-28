# Latte 条件嵌入层 MindSpore 实现

本文介绍如何使用 Mindspore 在 Ascend 910* 实现 Latte 中使用到的条件嵌入模块，主要代码参考自mindone套件 [opensora-pku/Latte](https://github.com/mindspore-lab/mindone/tree/master/examples/opensora_pku/opensora/models/diffusion/latte) 实现用到的 modules。

## 1. Latte 条件嵌入层简介

opensora-pku 使用 的 LatteT2V 网络以 transformer 为骨干，涉及 timestep 嵌入、位置编码、图像分块嵌入等模块。嵌入层对时间、空间、序列位置等信息进行编码学习。参考下图中的 LatteT2V 网络结构，涉及embedding 计算的方法或模块有：

* 位置编码函数（三角函数方法）
  * get_1d_sincos_pos_embed 
  * get_2d_sincos_pos_embed 

* 嵌入层
  * PatchEmbed  - 图块嵌入层
  * CombinedTimestepSizeEmbeddings - 时间-尺寸联合嵌入层
  * CaptionProjection - 文本嵌入映射层

![LatteT2V](./imgs/latteT2V.png)



## 2. Mindspore 代码实现

### 2.1 位置编码函数

* get_1d_sincos_pos_embed 
* get_2d_sincos_pos_embed 

位置编码函数在图块嵌入层 `PatchEmbed` 以及在LatteT2V网络计算时间位置嵌入 `temp_pos_embed` 变量时需要使用。定义位置编码函数，其中`mindone.diffusers` 中定义好的`get_2d_sincos_pos_embed`可直接复用。


```python
import numpy as np
from mindone.diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed

def get_1d_sincos_pos_embed(embed_dim, length, interpolation_scale=1.0, base_size=16):
    pos = np.arange(0, length)[:, None] / interpolation_scale
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return pos_embed
```



### 2.2 嵌入计算层

* PatchEmbed  - 图像块嵌入层

* CombinedTimestepSizeEmbeddings - 时间嵌入层
* CaptionProjection - 文本嵌入映射层



#### 2.2.1 PatchEmbed - 图像块嵌入层 

传统的卷积神经网络对图像处理使用像素级操作，而图像块嵌入 (Patch Embedding) 作为 transformer 结构在图像领域应用的开端，引入了新的特征表示方法：将输入图像分成小块，也就是 “patch”， 并将每个小块转化为低维度向量表示。代码实现时，`PatchEmbed`一般将图像分块后使用卷积后再平铺，这种做法相对直接使用线性映射层可提升效率；最后通过直接加上位置嵌入，使得模型注入`patches` 在图像中的位置信息。


```python
import mindspore as ms
from mindspore import nn, ops
from examples.opensora_pku.opensora.models.diffusion.latte.modules import LayerNorm

class PatchEmbed(nn.Cell):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, has_bias=bias)
        if layer_norm:
            self.norm = LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches**0.5), base_size=self.base_size, interpolation_scale=self.interpolation_scale
        )
        self.pos_embed = ms.Parameter(ms.Tensor(pos_embed).float().unsqueeze(0), requires_grad=False)

    def construct(self, latent):
        height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(start_dim=2).permute(0, 2, 1)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate positional embeddings if needed.
        # (For PixArt-Alpha: https://github.com/PixArt-alpha/PixArt-alpha/\
        # blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L162C151-L162C160)
        if self.height != height or self.width != width:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=(height, width),
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
            )
            pos_embed = ms.Tensor(pos_embed)
            pos_embed = pos_embed.float().unsqueeze(0)
        else:
            pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)
```



#### 2.2.2 CombinedTimestepSizeEmbeddings - 时间-尺寸联合嵌入

自适应归一化层`AdaLayerNormSingle` 需要使用的时间-空间尺寸联合嵌入作为输入，以控制归一化层的 scale 和 shift 参数。

时间-尺寸联合嵌入层关于时间嵌入部分的代码实现，可复用 `mindone.diffusers` 定义好的 sinusoidal timesteps embedding （即 `Timesteps`）和进一步的线性层映射操作（即 `TimestepEmbedding`)。

关于空间尺寸嵌入的计算，当 `use_additional_conditions` 设为 `True` 时，该模块会同时结合分辨率、纵横比等图像或视频的尺寸信息计算条件嵌入，与时间戳 `timestep` 共同控制 transformer blocks 的尺度和位移参数。该模块的关于尺寸信息的输入（ `resolution`, `aspect_ratio`, `batch_size`, `hidden_dtype` ）通过 `LatteT2V` 网络的 `added_cond_kwargs` 字典传入，见 [Latte 简介与 Mindspore 实现](./latte_implemented_from_scratch.md)。


```python
from mindone.diffusers.models.embeddings import TimestepEmbedding, Timesteps

class CombinedTimestepSizeEmbeddings(nn.Cell):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.use_additional_conditions = True
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def apply_condition(self, size: ms.Tensor, batch_size: int, embedder: nn.Cell):
        if size.ndim == 1:
            size = size[:, None]

        if size.shape[0] != batch_size:
            size = size.repeat_interleave(batch_size // size.shape[0], 1)
            if size.shape[0] != batch_size:
                raise ValueError(f"`batch_size` should be {size.shape[0]} but found {batch_size}.")

        current_batch_size, dims = size.shape[0], size.shape[1]
        size = size.reshape(-1)
        size_freq = self.additional_condition_proj(size).to(size.dtype)

        size_emb = embedder(size_freq)
        size_emb = size_emb.reshape(current_batch_size, dims * self.outdim)
        return size_emb

    def construct(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution = self.apply_condition(resolution, batch_size=batch_size, embedder=self.resolution_embedder)
            aspect_ratio = self.apply_condition(
                aspect_ratio, batch_size=batch_size, embedder=self.aspect_ratio_embedder
            )
            conditioning = timesteps_emb + ops.cat([resolution, aspect_ratio], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning
```



#### 2.2.3 CaptionProjection - 文本嵌入映射层

文生图、文生视频任务一般直接使用预训练的 text encoder 计算文本嵌入。例如 opensora-pku 训练时只训 Latte，T5 模型作为 text encoder 不参与训练。文本嵌入映射层作为 LatteT2V 的一部分，可以使得文本条件参与训练学习：T5 得到的 embeddings 经过可学习的映射层 `CaptionProjection` 再输入 transformer blocks。由于输入的 `caption`已经是 T5 的 embeddings 输出，该映射层没有对文本 tokens 的 dropout 操作。


```python
class CaptionProjection(nn.Cell):
    """Projects caption embeddings.
    """

    def __init__(self, in_features, hidden_size, num_tokens=120):
        super().__init__()
        self.linear_1 = nn.Dense(in_features, hidden_size)
        self.act_1 = nn.GELU(True)
        self.linear_2 = nn.Dense(hidden_size, hidden_size)

    def construct(self, caption, force_drop_ids=None):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
```

#### 


## 3. 扩展阅读

本文基于 MindSpore + Ascend910* 介绍了`mindone`仓的 opensora-pku/LatteT2V 网络涉及的条件嵌入计算，LatteT2V 完整代码可参考 [examples/opensora_pku/opensora/models/diffusion/latte](https://github.com/mindspore-lab/mindone/tree/master/examples/opensora_pku/opensora/models/diffusion/latte)。