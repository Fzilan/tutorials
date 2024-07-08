# MindSpore 文生图训练自定义数据集构造

本文以 mindone 套件 sdxl 的训练实现为例，介绍如何基于 `mindspore.dataset` 模块构造一般文生图训练使用的数据集与数据加载器。MindSpore 自定义数据集的构建流程参考自 MindSpore 官网 [数据集 Dataset](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html) 教程，代码参考自 mindone 套件 [sdxl data](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_xl/gm/data) 部分的实现。



## 1. 数据准备

SD 系列文生图扩散模型的训练数据集一般为图文对，即图片与文本描述。以 mindone 套件的 [sd_v2](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_v2)， [sdxl](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_xl) 实现为例, 图文对训练数据集的存放格式如下：

```text
data_path
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── img_txt.csv
```

其中 `img_txt.csv` 即为图片-文本描述的标注文件，格式如下：

```text
dir,text
img1.jpg,a cartoon character with a potted plant on his head
img2.jpg,a drawing of a green pokemon with red eyes
img3.jpg,a red and white ball with an angry look on its face
```

第一列指的是图片相对 `data_path` 的存放位置，第二列是图片对应的文本描述。

方便起见，mindone 仓把宝可梦数据集以及中国画数据集准备成上述存放格式，并上传到 [openi dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets) 网站，下载 zip 文件放到本地路径下解压可获取。用户也可以根据自己的需求准备其他数据集。

- [pokemon-blip-caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 833 pokemon-style images with BLIP-generated captions.
- [Chinese-art blip caption dataset](https://openi.pcl.ac.cn/jasonhuang/mindone/datasets), containing 100 chinese art-style images with BLIP-generated captions.


本案例以 Chinese-art 数据集为例，解压后存放于 `./datasets/chinese_art_blip` 路径下。

定义方法 `list_image_files_captions_recursively`读取制定路径下的数据，该方法可作为下面的可访问数据集类的静态方法之一：


```python
import os
import pandas as pd
def list_image_files_captions_recursively(data_path):
    anno_dir = data_path
    anno_list = sorted(
        [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
    )
    db_list = [pd.read_csv(f) for f in anno_list]
    all_images = []
    all_captions = []
    for db in db_list:
        all_images.extend(list(db["dir"]))
        all_captions.extend(list(db["text"]))
    assert len(all_images) == len(all_captions)
    all_images = [os.path.join(data_path, f) for f in all_images]

    return all_images, all_captions
```


```python
data_path = "./datasets/chinese_art_blip/train"
all_images, all_captions = list_image_files_captions_recursively(data_path)

print("打印一个样本：")
print(all_images[0])
print(all_captions[0])
```

    打印一个样本：
    ./datasets/chinese_art_blip/train/640.jpeg
    a painting of two birds perched on a tree branch


## 2. 构造可随机访问的数据集

按照官网自定义数据集的教程构造可随机访问数据集 `Text2ImageDataset`。 数据集需要实现 `__getitem__` 和 `__len__` 方法，表示可以通过索引/键直接访问对应位置的数据样本。

### 2.1 数据集初始化

参数：
- `data_path` - 存放数据的路径
- `target_size` - 图像的目标尺寸，sdxl 使用 (1024, 1024)
- `transforms` - 数据转换 mapper 配置， 在 mindone sdxl 实现中通过 yaml 配置传入
- `batched_transforms` - 批数据转换 mapper 配置，在 mindone sdxl 实现中通过 yaml 配置传入
- `tokenizer` - 通过传入 tokenizer 可在数据加载时提前计算 tokens
- `token_nums` - token 数量，假设加载数据时直接加载 token， sdxl token_nums=5
- `filter_small_size` - 是否过滤小尺寸的图像
- `image_filter_size` - 图像过滤尺寸，h w 小于该尺寸的则过滤
- `multi_aspect` - sdxl 多尺度训练配置  # for multi_aspect
- `seed` - for multi_aspect
- `prompt_empty_probability` - 空文本概率，可理解为描述文本的 dropout 操作

初始化方法定义如下，主要实现读取数据路径下的数据，获取图像的相对路径列表与文本列表，以及初始化配置中的转换方法：


```python
import os
import random
import imagesize
import numpy as np

# clone mindone, cd examples/stable_diffusion_xl
from gm.data.util import _is_valid_text_input
from gm.util import instantiate_from_config

class Text2ImageDataset:
    def __init__(
        self,
        data_path,
        target_size=(1024, 1024),
        transforms=None,
        batched_transforms=None,
        tokenizer=None,
        token_nums=None,
        image_filter_size=0,
        filter_small_size=False,
        multi_aspect=None,  # for multi_aspect
        seed=42,  # for multi_aspect
        prompt_empty_probability=0.0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.token_nums = token_nums
        self.dataset_column_names = ["samples"]
        if self.tokenizer is None:
            self.dataset_output_column_names = self.dataset_column_names
        else:
            assert token_nums is not None and token_nums > 0
            self.dataset_output_column_names = [
                "image",
            ] + [f"token{i}" for i in range(token_nums)]

        self.target_size = [target_size, target_size] if isinstance(target_size, int) else target_size
        self.filter_small_size = filter_small_size

        self.multi_aspect = list(multi_aspect) if multi_aspect is not None else None
        if self.multi_aspect and len(self.multi_aspect) > 10:
            random.seed(seed)
            self.multi_aspect = random.sample(self.multi_aspect, 10)
            print(f"Text2ImageDataset: modify multi_aspect sizes to {self.multi_aspect}")

        self.seed = seed
        self.prompt_empty_probability = prompt_empty_probability

        all_images, all_captions = self.list_image_files_captions_recursively(data_path)
        if filter_small_size:
            all_images, all_captions = self.filter_small_image(all_images, all_captions, image_filter_size)
        self.local_images = all_images
        self.local_captions = all_captions

        self.transforms = []
        if transforms:
            for i, trans_config in enumerate(transforms):
                # Mapper
                trans = instantiate_from_config(trans_config)
                self.transforms.append(trans)
                print(f"Adding mapper {trans.__class__.__name__} as transform #{i} " f"to the datapipeline")

        self.batched_transforms = []
        if batched_transforms:
            for i, bs_trans_config in enumerate(batched_transforms):
                # Mapper
                bs_trans = instantiate_from_config(bs_trans_config)
                self.batched_transforms.append(bs_trans)
                print(
                    f"Adding batch mapper {bs_trans.__class__.__name__} as batch transform #{i} " f"to the datapipeline"
                )
    @staticmethod
    def list_image_files_captions_recursively(data_path):
        anno_dir = data_path
        anno_list = sorted(
            [os.path.join(anno_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(anno_dir)))]
        )
        db_list = [pd.read_csv(f) for f in anno_list]
        all_images = []
        all_captions = []
        for db in db_list:
            all_images.extend(list(db["dir"]))
            all_captions.extend(list(db["text"]))
        assert len(all_images) == len(all_captions)
        all_images = [os.path.join(data_path, f) for f in all_images]

        return all_images, all_captions

    @staticmethod
    def filter_small_image(all_images, all_captions, image_filter_size):
        filted_images = []
        filted_captions = []
        for image, caption in zip(all_images, all_captions):
            w, h = imagesize.get(image)
            if min(w, h) < image_filter_size:
                print(f"The size of image {image}: {w}x{h} < `image_filter_size` and excluded from training.")
                continue
            else:
                filted_images.append(image)
                filted_captions.append(caption)
        return filted_images, filted_captions

```

### 2.2 可随机访问方法

实现 `__getitem__` 和 `__len__` 方法，通过索引访问数据样本。

- `__getitem__` 

    输入：索引
    输出：样本字典，sdxl 字典的 key 为：

    - image - 图像数据，np.array 格式
    - txt - 文本描述，np.array 格式
    - original_size_as_tuple - 图像的原始尺寸 (h w)
    - target_size_as_tuple - 图像的目标尺寸 (h w)，sdxl 为 10
    - crop_coords_top_left - 裁剪的左上方坐标点
    - aesthetic_score - 艺术分数

    上述数据中，除了图像与文本信息，sdxl 的训练还使用了三个图像尺寸相关的信息。生成式模型中典型的图像预处理方式为先调整图像尺寸，使得最短边与目标尺寸匹配，然后再沿较长边对图像进行随机裁剪或者中心裁剪。被裁剪部分的图像特征丢失，则模型生成阶段可能会出现不符合训练数据分布的特征。sdxl 使用图像裁剪参数条件化策略，记录裁剪后的左上角目标，与原始图像的尺寸一起作为额外的条件嵌入训练模型，使模型认识裁剪。
    
    `aesthetic_score` 直接赋予了一个值，事实上这个信息在 mindone sdxl 训练中暂时没有被使用。

    `__getitem__` 方法根据初始化参数给定的 `transforms` 对单样本进行了数据转换再输出。

- `__len__` 

    输出：数据集样本数量



```python
from PIL import Image
from PIL.ImageOps import exif_transpose

class Text2ImageDataset:
    ...

    def __getitem__(self, idx):
        # images preprocess
        image_path = self.local_images[idx]
        image = Image.open(image_path)
        original_size = [image.height, image.width]
        image = self.shrink_pixels(image, self.max_pixels)
        image = exif_transpose(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # caption preprocess
        caption = (
            ""
            if self.prompt_empty_probability and random.random() < self.prompt_empty_probability
            else self.local_captions[idx]
        )

        if not _is_valid_text_input(caption):
            print(
                f"WARNING: text input must of type `str`, but got type: {type(caption)}, caption: {caption}", flush=True
            )

            caption = str(caption)

            if _is_valid_text_input(caption):
                print("WARNING: convert caption type to string success.", flush=True)
            else:
                caption = " "
                print("WARNING: convert caption type to string fail, set caption to ` `.", flush=True)

        caption = np.array(caption)

        sample = {
            "image": image,
            "txt": caption,
            "original_size_as_tuple": np.array(original_size, np.int32),  # original h, original w
            "target_size_as_tuple": np.array([self.target_size[0], self.target_size[1]]),  # target h, target w
            "crop_coords_top_left": np.array([0, 0]),  # crop top, crop left
            "aesthetic_score": np.array([
                    6.0,
                ]
            ),
        }

        for trans in self.transforms:
            sample = trans(sample)

        return sample

    def __len__(self):
        return len(self.local_images)
```

### 2.3 批处理的可调用对象

在下面第 3 小节的数据加载器中，可随机访问的自定义数据集 `Text2ImageDataset` 加载到 `mindspore.dataset.GeneratorDataset` 后，可以使用 `batch` 方法 将数据集中连续 batch_size 条数据组合为一个批数据，并可以通过可选参数 `per_batch_map` 指定组合前要进行的预处理操作。

`per_batch_map` 参数传的是一个可调用对象，具体可参考框架接口文档 [mindspore.dataset.Dataset.batch](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html?highlight=batch#mindspore.dataset.Dataset.batch)。 我们可以在定义 `Text2ImageDataset` 时先实现这个可调用对象，代码示例如下：


```python
class Text2ImageDataset:
    ...
    def batch_collate_fn(self, samples, batch_info):
        new_size = self.target_size
        if self.multi_aspect:
            raise NotImplementedError("please refer to mindone repo")
        for bs_trans in self.batched_transforms:
            samples = bs_trans(samples, target_size=new_size)
        batch_samples = {k: [] for k in samples[0]}
        for s in samples:
            for k in s:
                batch_samples[k].append(s[k])
        data = {k: (np.stack(v, 0) if isinstance(v[0], np.ndarray) else v) for k, v in batch_samples.items()}
        if self.tokenizer:
            raise NotImplementedError("please refer to mindone repo")
        else:
            outs = data
        return outs
```

备注：完整的预处理操作如 tokenize 、多尺度处理等可参考 mindone 仓实现。

### 2.4 实例化结果

实例化定义好的图文数据集，数据路径使用 chinese_art_blip 训练数据集所在路径。下面代码中，`Text2ImageDataset` 的 `transforms` 直接使用 mindone 库的定义好的一些数据转换 mapper。实例化数据集并打印结果：


```python
import math
import multiprocessing
import mindspore.dataset as de

# clone mindone, cd examples/stable_diffusion_xl
from gm.util import get_obj_from_str

data_path = './datasets/chinese_art_blip/train'
dataset_config = {
    'target': 'gm.data.dataset.Text2ImageDataset', 
    'params': 
        {
         'target_size': 1024, 
         'transforms':
            [
             {'target': 'gm.data.mappers.Resize', 'params': {'size': 1024, 'interpolation': 3}}, 
             {'target': 'gm.data.mappers.Rescaler', 'params': {'isfloat': False}}, 
             {'target': 'gm.data.mappers.AddOriginalImageSizeAsTupleAndCropToSquare'}, 
             {'target': 'gm.data.mappers.Transpose', 'params': {'type': 'hwc2chw'}}
            ]
        }
    }
dataset = get_obj_from_str(dataset_config["target"])(
    data_path=data_path,
    **dataset_config.get("params", dict()),
)
```

    Adding mapper Resize as transform #0 to the datapipeline
    Adding mapper Rescaler as transform #1 to the datapipeline
    Adding mapper AddOriginalImageSizeAsTupleAndCropToSquare as transform #2 to the datapipeline
    Adding mapper Transpose as transform #3 to the datapipeline


查看数据集大小：


```python
len(dataset)
```




    80



给定索引获取对应样本：


```python
dataset[50]
```




    {'image': array([[[0.82745098, 0.82745098, 0.82745098, ..., 0.88235294,
              0.88235294, 0.8745098 ],
             ...,
             [0.89019608, 0.89019608, 0.89803922, ..., 0.91372549,
              0.91372549, 0.92156863]],
             ...,
            [[0.70980392, 0.70980392, 0.70980392, ..., 0.74117647,
              0.74117647, 0.73333333],
             ...,
             [0.77254902, 0.77254902, 0.78039216, ..., 0.79607843,
              0.79607843, 0.80392157]]]),
     'txt': array('a drawing of a snowy landscape with a house in the distance and trees in the foreground',
           dtype='<U87'),
     'original_size_as_tuple': array([494, 700], dtype=int32),
     'target_size_as_tuple': array([1024, 1024]),
     'crop_coords_top_left': array([  0, 359]),
     'aesthetic_score': array([6.])}



## 3. dataloader

可随机访问数据集定义好，可通过 [`mindspore.dataset.GeneratorDataset`](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html) 接口实现数据集加载。

下面定义 create_loader 接口，加载 `Text2ImageDataset` 至 `GeneratorDataset`，并通过给 `.batch` 接口传递 2.3 小节定义的批处理可调用对象 `Text2ImageDataset.batch_collate_fn` 做批处理。接口的主要入参有：


- 初始化 `Text2ImageDataset` 使用的参数，此处选择3个暴露给 create_loader 接口，其他由 `dataset_config` 配置，可参考 2.4 小节的示例。
    - `data_path` - 数据路径
    - `tokenizer` 
    - `token_nums` 
    - `dataset_config`

- 加载至 `GeneratorDataset` 
    - `rank_size` - 用于判断是否多卡训练，是则在 `GeneratorDataset` 指定分布式训练时将数据集进行划分的分片数
    - `rank` - 多卡训练时在 `GeneratorDataset` 指定分布式训练时使用的分片ID号
    - `python_multiprocessing` - 在 `GeneratorDataset` 指定是否启用Python多进程模式加速运算
    - `shuffle` - 是否混洗数据集
- batch 
    - `drop_remainder` - 当最后一个批处理数据包含的数据条目小于 batch_size 时，是否将该批处理丢弃。
    - `per_batch_size` - 单批数据条目


```python
def create_loader(
    data_path,
    rank=0,
    rank_size=1,
    *,
    dataset_config,
    per_batch_size,
    total_step=1000,
    num_epochs=0,
    num_parallel_workers=8,
    shuffle=True,
    drop_remainder=True,
    python_multiprocessing=False,
    tokenizer=None,
    token_nums=None,
):
    r"""Creates dataloader.

    Applies operations such as transform and batch to the `ms.dataset.Dataset` object
    created by the `create_dataset` function to get the dataloader.

    Returns:
        BatchDataset, dataset batched.
    """
    
    dataset = get_obj_from_str(dataset_config["target"])(
        data_path=data_path,
        tokenizer=tokenizer,
        token_nums=token_nums,
        **dataset_config.get("params", dict()),
    )
    batch_collate_fn, dataset_column_names, dataset_output_column_names = (
        dataset.collate_fn,
        dataset.dataset_column_names,
        dataset.dataset_output_column_names,
    )
    dataset_size = len(dataset)
    num_step_per_epoch = dataset_size // (per_batch_size * rank_size)
    epoch_size = num_epochs if num_epochs else math.ceil(total_step / num_step_per_epoch)

    de.config.set_seed(1236517205 + rank)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(int(cores / min(rank_size, 8)), num_parallel_workers)
    print(f"Dataloader num parallel workers: [{num_parallel_workers}]")
    if rank_size > 1:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
            num_shards=rank_size,
            shard_id=rank,
        )
    else:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(32, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
        )
    ds = ds.batch(
        per_batch_size,
        per_batch_map=batch_collate_fn,
        input_columns=dataset_column_names,
        output_columns=dataset_output_column_names,
        num_parallel_workers=min(8, num_parallel_workers),
        drop_remainder=drop_remainder,
    )
    ds = ds.repeat(epoch_size)

    return ds

```
调用 `create_loader` 接口：

```python
data_path = '/Users/fanzhilan/mindspore/mindone/examples/stable_diffusion_xl/gm/data/datasets/chinese_art_blip/train'
train_dataset_loader= create_loader(
    data_path=data_path, # defined in section 2.4
    per_batch_size=1,
    dataset_config=dataset_config, # defined in section 2.4
)
```

    Adding mapper Resize as transform #0 to the datapipeline
    Adding mapper Rescaler as transform #1 to the datapipeline
    Adding mapper AddOriginalImageSizeAsTupleAndCropToSquare as transform #2 to the datapipeline
    Adding mapper Transpose as transform #3 to the datapipeline
    Dataloader num parallel workers: [8]

使用 `create_tuple_iterator` 方法迭代获取 MindSpore 的数据对象：基于数据集对象创建迭代器，输出数据为 numpy.ndarray 组成的列表：


```python
total_step = train_dataset_loader.get_dataset_size()
loader = train_dataset_loader.create_tuple_iterator(output_numpy=True, num_epochs=1)
for i, data in enumerate(loader):
    print(f"Step {i + 1}/{total_step}....")
    print(data[0])
    break
```

    Step 1/1040....
    {'image': array([[[[1.        , 1.        , 1.        , ..., 0.43529412,
              0.27843137, 0.20784314],
             ...,
             [0.89019608, 0.89019608, 0.89803922, ..., 0.71764706,
              0.74117647, 0.76470588]],
             ...,
    
            [[1.        , 1.        , 1.        , ..., 0.41960784,
              0.2627451 , 0.19215686],
             ...,
             [0.80392157, 0.81176471, 0.81960784, ..., 0.70196078,
              0.73333333, 0.76470588]]]]), 
    'txt': array(['a painting of a mountain with trees on it and clouds in the sky above it'], dtype='<U72'), 
    'original_size_as_tuple': array([[721, 680]], dtype=int32), 
    'target_size_as_tuple': array([[1024, 1024]]), 
    'crop_coords_top_left': array([[55,  0]]), 
    'aesthetic_score': array([[6.]])}


## 4. 扩展阅读

MindSpore 数据集相关官网教程：
- [入门教程 - 数据集 Dataset](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/dataset.html)
- [入门教程 - 数据变换 Transforms](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/transforms.html)
- [mindspore.dataset: 数据处理 pipeline 文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.dataset.html#)
- [MindSpore 与 torch 网络搭建对比之数据处理](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/dataset.html)

mindone 仓 sdxl 代码实现：
- [mindone sdxl implement and readme](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_xl)
- [训练数据集构建 - examples/stable_diffusion_xl/gm/data](https://github.com/mindspore-lab/mindone/tree/master/examples/stable_diffusion_xl/gm/data)