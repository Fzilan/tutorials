### MindSpore 静态图训练流程简介

#### 1、我们尝试构建一个简单的 `net with loss`

这一步是对 `整体模型` 及 `损失函数计算` 进行封装；

备注：下面用一个dense层作为示例，实际运行的时候需要替换为真实的训练模型；

```python
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor

class NetWithLoss(nn.Cell):
    def __init__(self, network: nn.Cell):
        super().__init__()
        self.network = network
        
    def construct(self, x: ms.Tensor, target: ms.Tensor):
        out = self.network(x)
        loss = ops.mean((out - target) ** 2)
        return loss

net = nn.Dense(in_channels=128, out_channels=256)
net_with_loss = NetWithLoss(net)
x = Tensor(np.random.randn(2, 128), ms.float32)
target = Tensor(1.0, ms.float32)
loss = net_with_loss(x, target)
```

```shell
// 补充一个打印结果
```

#### 2、我们尝试构造一个用于一步训练的cell，并启动训练

这一步会构建 `train one step` 过程，这里面是对整网训练一个step的封装，包括 `网络前向`、`loss计算`、`反向传播`、`优化器更新`；

备注：下面用一个dense层作为示例，实际运行的时候需要替换为真实的训练模型；

```python
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.ops import functional as F

class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        net_with_loss,
        optimizer,
    ):
        super().__init__()
        self.net_with_loss = net_with_loss
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(net_with_loss, grad_position=None, weights=optimizer.parameters)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

net = nn.Dense(in_channels=128, out_channels=256)
net_with_loss = NetWithLoss(net)  # from section 3.1
optimizer = nn.AdamWeightDecay(net_with_loss.trainable_params(), learning_rate=1e-5)
train_one_step = TrainOneStepCell(net_with_loss, optimizer)
x = Tensor(np.random.randn(2, 128), ms.float32)
target = Tensor(1.0, ms.float32)
for i in range(10):
    loss = train_one_step(x, target)
    print(f"Step: {i}, loss: {loss}")
```

```shell
// 补充一个打印结果
```
