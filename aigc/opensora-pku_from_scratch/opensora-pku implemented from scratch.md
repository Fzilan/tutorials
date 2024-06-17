# OpenSora-PKU 从零到一实现

本文介绍如何使用 Mindspore 在 Ascend 910* 上从零到一实现 OpenSora-PKU 视频生成模型(静态图)，主要代码参考自mindone套件中[opensora-pku](https://github.com/mindspore-lab/mindone/tree/master/examples/opensora_pku)实现。

### 1、基本训练流程介绍

opensora-pku v1.0 整体结构使用 `T5` 作为默认文本编码器，`VideoCausalVAE` 作为默认视频压缩模型 以及 `Latte` 作为噪声预测模型，
损失函数默认使用 `MSE loss`，训练过程采用图像、视频混合训练策略，整体流程如下图所示：

![整体流程图](./imgs/img1.png)

### 2、主要模块搭建

#### 2.1 vae 模块搭建

- [VideoCausalVAE 简介与 MindSpore 实现]()

```python
import numpy as np
import mindspore as ms
from mindspore import Tensor
from .opensora.models.ae.videobase.causal_vae import CausalVAEModelWrapper

vae = CausalVAEModelWrapper(
    model_config="./causal_vae_488.yaml",
    model_path="LanguageBind/Open-Sora-Plan-v1.0.0",
    subfolder="vae"
)

ori_video = Tensor(np.random.randn(...))
print(f"original video input shape: {ori_video.shape}")
print(f"original video value: {ori_video[:, :, :20]}")

latent = vae.encode(ori_video)
print(f"latent.shape: {latent.shape}")
print(f"latent value: {latent[:, :, :20]}")

reconstruct_video = vae.decode(latent)
print(f"reconstruct video input shape: {reconstruct_video.shape}")
print(f"reconstruct video value: {reconstruct_video[:, :, :20]}")
```

```shell
// 补充打印结果
```

#### 2.2 t5 模块搭建

- [T5-XXL 简介与 MindSpore 实现]()

```python
import mindspore as ms
from mindspore import Tensor
from .opensora.models.text_encoder.t5 import T5Embedder
from .opensora.xxx import xxx as tokenizer

t5 = T5Embedder(
    dir_or_name="DeepFloyd/t5-v1_1-xxl",
    cache_dir="./",
    model_max_length=300,
)

text = "a cute dog"
tokens = tokenizer(text)
tokens = Tensor(tokens, ms.float32)
text_embedding = t5(tokens)

print(f"text: {text}")
print(f"tokens shape: {tokens}")
print(f"tokens: {tokens.view(-1)[:20]}")
print(f"text embedding shape: {text_embedding.shape}")
print(f"text embedding: {text_embedding.view(-1)[:20]}")
```

```shell
// 补充打印结果
```

#### 2.3 latte 模块搭建

- [Latte 简介与 MindSpore 实现]()

```python
from .opensora.models.diffusion.latte.modeling_latte import LatteT2V_XL_122

latte_model = LatteT2V_XL_122(
    in_channels=4,
    out_channels=4 * 2,
    ...
)
```

```shell
// 补充打印结果
```

#### 2.4 diffusion 模块搭建

- [diffusion 过程定义]()

```python
from .opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion

diffusion = create_diffusion(timestep_respacing="")
```

```shell
// 补充打印结果
```

### 3、数据迭代器搭建

[视频数据集创建]()

```python
from .opensora.dataset.t2v_dataset import TextVideoDataset, create_dataloader

dataset = TextVideoDataset(...)
loader = create_dataloader(dataset)  # 使用minddata进行封装

print(f"dataset_size: {loader.get_dataset_size()}")
index = 0
for i, data in enumerate(loader):
    print(data)
    if i > 10: break
```

```shell
// 补充一个打印结果
```

### 4、训练流程搭建

#### 4.1 [MindSpore 静态图训练流程小示例](./docs/MindSpore static graph training process introduction.md)

#### 4.2 opensora-pku 1.0 训练流程 (用于流程说明，代码不可运行)

```python
from mindspore import Model
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from .opensora.models.ae.videobase.causal_vae import CausalVAEModelWrapper
from .opensora.models.text_encoder.t5 import T5Embedder
from .opensora.models.diffusion.latte.modeling_latte import LatteT2V_XL_122
from .opensora.models.diffusion.diffusion import create_diffusion_T as create_diffusion
from .opensora.models.diffusion.latte.net_with_loss import DiffusionWithLoss

latte = LatteT2V_XL_122(...)
vae = CausalVAEModelWrapper(...)
text_encoder = T5Embedder(...)
diffusion = create_diffusion(timestep_respacing="")
net_with_loss = DiffusionWithLoss(
        latte,
        diffusion,
        vae=vae,
        text_encoder=text_encoder,
        ...
)

optimizer = create_optimizer(net_with_loss.trainable_params())

train_one_setp = TrainOneStepWrapper(
    net_with_loss,
    optimizer=optimizer,
    ...
)

# MindSpore 内置模型封装
model = Model(train_one_setp)

# 启动训练
model.train(
    epoch=1,
    train_dataset=dataset,
    callbacks=callback,
    ...
)
```







