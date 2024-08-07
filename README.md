# Tutorials

### AIGC/文生图

- 文生图数据集构建
    - [x] [MindSpore 文生图训练数据集构造](./aigc/sdxl_implemented_from_scratch/create_text2image_datasets.md)
    - [ ] [MindSpore 文生图 dreambooth 微调数据集构造]
    - [ ] [基于 huggingface `datasets` 库构造 MindSpore 数据集]

- [ ] [SDXL 从零到一实现]
    - [ ] [SDXL - VAE模块构建]
    - [ ] [SDXL - Conditioner块构建]
    - [ ] [SDXL - Clip模块构建]
    - [ ] [SDXL - OpenClip模块构建]
    - [ ] [SDXL - Unet模块构建]
    - 训练相关
      - [ ] [SDXL LoRA训练构建]
      - [ ] [SDXL Dreambooth训练构建]
    - 推理相关
      - [x] [SDXL 推理流程介绍与 MindSpore 实现](./aigc/sdxl_implemented_from_scratch/sdxl-infer.md) 
      - [x] [Euler 采样器介绍与 Mindspore 实现](./aigc/sdxl_implemented_from_scratch/sampler-implement.md) 

- SD-VAE 系列
    - [ ] [2D-VAE]
    - [ ] [VQ-VAE]

### AIGC/文生视频

- [x] [OpenSora-PKU 从零到一实现](./aigc/opensora-pku_from_scratch/opensora-pku%20implemented%20from%20scratch.md)
    - [ ] [VideoCausalVAE 简介与 MindSpore 实现]
    - [ ] [T5-XXL 简介与 MindSpore 实现]
    - [x] [Latte 简介与 MindSpore 实现](./aigc/opensora-pku_from_scratch/latte_implemented_from_scratch.md)
    - [x] [Latte 条件嵌入层 MindSpore 实现](./aigc/opensora-pku_from_scratch/latte_embedding_modules_implement.md)
    - [x] [自适应归一化层 MindSpore 实现](./aigc/opensora-pku_from_scratch/latte_adalayernorm_implement.md)
    - [x] [Latte 多头注意力模块的 MindSpore 实现](./aigc/opensora-pku_from_scratch/latte_mha_implement.md)
    - [x] [Latte BasicTransformerBlock MindSpore 实现](./aigc/opensora-pku_from_scratch/latte_transformerblock_implement.md)
    - diffusion 过程
      - [ ] [diffusion 过程定义]
      - [ ] [loss function 实现]
      - [ ] [DDPM 训练过程简介与 MindSpore 实现]
    - 训练优化
      - [x] [vae tiling 实现](./aigc/opensora-pku_from_scratch/docs/vae_tiling_implement.md)
      - [ ] [text embedding cache 实现]
    - 推理相关
      - [ ] [opensora-pku 推理流程介绍与MindSpore实现]
      - [ ] [PNDM scheduler 介绍与MindSpore实现]
      - [ ] [DDIM scheduler 介绍与MindSpore实现]

### AIGC/文生音频

- [MusicGen 介绍与推理实现]

### API 接口介绍

- [nn.Conv3D 接口实践]