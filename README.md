# CodonTransformer_NPU
CodonTransformer model adapted for Ascend 910A NPU
# 昇腾910A下CodonTransformer模型基于Lightning框架的NPU迁移训练指南
## 一、项目概述
本项目聚焦于将CodonTransformer模型迁移至昇腾910A芯片上进行训练，借助PyTorch Lightning框架实现高效开发与部署。在迁移过程中，攻克了诸多框架与NPU设备的适配难题，旨在为生物信息学等相关领域提供基于国产算力平台的优化模型训练方案。

## 二、环境配置
确保以下依赖库已正确安装，版本如下：
- `Python`：3.10.0
- `pytorch-lightning`：1.9.5
- `torch`：2.1.0
- `torch-npu`：2.1.0.post12
- `torchvision`：0.16.0

## 三、关键资源链接
- **原始代码**：[https://github.com/Adibvafa/CodonTransformer](https://github.com/Adibvafa/CodonTransformer)
- **模型地址**：[https://huggingface.co/adibvafa/CodonTransformer](https://huggingface.co/adibvafa/CodonTransformer)
- **数据集地址**：[https://huggingface.co/datasets/adibvafa/CodonTransformer](https://huggingface.co/datasets/adibvafa/CodonTransformer)
- **NPU适配参考代码**：[https://gitee.com/xiaoloongfang/lightning_npu](https://gitee.com/xiaoloongfang/lightning_npu)
- **完整参考代码**：[git clone https://gitee.com/zhou-xingzi/codon-transformer_-npu.git](https://gitee.com/zhou-xingzi/codon-transformer_-npu.git)

## 四、核心问题与解决方案
### 4.1 ImportError: cannot import name 'torch_device_guard'
- **错误位置**：`lightning_npu/core/maxins/device_dtype_mixin.py`
- **解决方案**：将导入语句从`from torch_npu.utils.tensor_methods import torch_device_guard`修改为`from torch_npu.utils.tensor_methods import npu_device_patch`。这是由于NPU相关工具函数路径调整，需要适配新的导入方式。

### 4.2 ImportError: cannot import name '_DEVICE'
- **错误位置**：`lightning_npu/accelerators/npu.py`
- **解决方案**：把`from pytorch_lightning.utilities.types import _DEVICE`修改为`import torch_npu`。原因是`_DEVICE`在新的代码结构中不再适用，直接引入`torch_npu`可满足后续对NPU设备操作的需求。

### 4.3 ImportError: cannot import name '_PATH'
- **错误位置**：`lightning_npu/accelerators/npu.py`
- **解决方案**：将`from pytorch_lightning.utilities.types import _PATH`修改为`from typing import Union`。因为`_PATH`并非必要的导入项，`typing.Union`用于处理类型提示，可保证代码类型相关功能正常运行。

### 4.4 TypeError: Can't instantiate abstract class NPUAccelerator
- **错误原因**：`NPUAccelerator`类缺少抽象方法，导致无法实例化。
- **解决方案**：在`NPUAccelerator`类中补充以下方法：
```python
def setup_device(self, device):
    torch_npu.npu.set_device(device)
    
def teardown(self):
    pass
```
这些方法分别用于设置NPU设备以及资源清理（此处`teardown`暂时为空实现，可根据实际需要后续完善）。

### 4.5 Precision '16-mixed' is invalid
- **解决方案**：取消混合精度训练，移除相关配置并采用“bf16”精度进行训练。因为当前NPU环境对“16-mixed”混合精度支持不佳，“bf16”精度既保证计算效率又能适配NPU硬件特性。

### 4.6 AttributeError: module 'torch_npu.npu' has no attribute 'native_device'
- **错误位置**：`lightning_npu/core/maxins/device_dtype_mixin.py`
- **解决方案**：删除`native_device = str(self.device).replace("npu", torch_npu.npu.native_device)`，替换为`torch_npu.npu.set_device(device_str)`。原因是`torch_npu.npu`下不存在`native_device`属性，直接设置设备更为简洁有效。

### 4.7 ValueError: Expected positive integer total_steps, but got -1
- **错误原因**：学习率调度器`total_steps`计算错误。
- **解决方案**：手动计算`steps_per_epoch`和`total_steps`：
```python
steps_per_epoch = self.samples_count // self.batch_size
total_steps = steps_per_epoch * self.trainer.max_epochs
```
通过正确的样本数量和批次大小计算每个epoch的步数，进而得出总的训练步数，确保学习率调度器正常工作。

### 4.8 RuntimeError: Cannot re-initialize NPU in forked subprocess
- **解决方案**：训练前将数据置于CPU，训练时加载到NPU，避免多进程直接操作NPU张量。这是因为NPU在多进程场景下的初始化机制限制，采用CPU中转数据可有效规避该问题。

## 五、训练命令示例
使用如下命令启动训练：
```bash
python finetune.py \
--dataset_dir 'dataset/training_data.jsonl' \
--checkpoint_dir './ckpt' \
--checkpoint_filename 'finetune.ckpt' \
--batch_size 6 \
--max_epochs 15 \
--num_workers 5 \
--accumulate_grad_batches 1 \
--num_gpus 1 \
--learning_rate 0.00005 \
--warmup_fraction 0.1 \
--save_every_n_steps 512 \
--seed 23
```
各参数解释如下：
- `--dataset_dir`：指定训练数据集路径。
- `--checkpoint_dir`：设置模型检查点保存目录。
- `--checkpoint_filename`：定义检查点文件名。
- `--batch_size`：训练批次大小。
- `--max_epochs`：最大训练轮数。
- `--num_workers`：数据加载器使用的进程数。
- `--accumulate_grad_batches`：梯度累积的批次数量。
- `--num_gpus`：使用的GPU数量（在NPU场景下，此参数实际指定使用的NPU设备数量）。
- `--learning_rate`：学习率。
- `--warmup_fraction`：学习率warmup阶段的比例。
- `--save_every_n_steps`：每多少步保存一次检查点。
- `--seed`：随机数种子，用于实验可重复性。

## 六、训练效果展示
### 6.1 模型参数统计
```
| Name      | Type               | Params
---------------------------------------------
0 | model | BigBirdForMaskedLM | 89.6 M
---------------------------------------------
89.6 M    Trainable params
0         Non-trainable params
89.6 M    Total params
358.318   Total estimated model params size (MB)
```
模型主体为`BigBirdForMaskedLM`，总参数量89.6M，其中可训练参数89.6M ，预估模型大小358.318MB 。

### 6.2 NPU资源使用
```
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 48988         | python                   | 13330                   |
+===========================+===============+====================================================+
```
训练过程中，NPU芯片0上运行着进程ID为48988的Python程序，占用内存13330MB ，表明NPU资源得到合理利用。

### 6.3 训练日志片段
```
Epoch 0:  12%|█████████▌                                                                       | 2/17 [14:17<1:47:13, 428.87s/it, loss=0.977, v_num=3, lr=2e-6]
```
从日志中可看出，在第0个epoch，已完成2/17个训练步骤，每个步骤耗时428.87秒，当前损失值为0.977 ，学习率为2e-6 ，训练进度和关键指标一目了然。

## 七、总结建议
1. **NPU迁移要点**：在NPU迁移过程中，设备初始化、多进程管理及数据加载流程是关键环节，需重点关注并优化。例如，按前文所述，正确设置NPU设备，避免多进程直接操作NPU张量等。
2. **框架适配策略**：当遇到框架适配问题时，优先查阅官方适配库（如`lightning_npu`）。这些库经过官方验证与优化，能快速解决大部分兼容性问题。
3. **混合精度调试**：在NPU上进行混合精度训练时需谨慎配置。建议先从纯精度模式开始调试，待模型稳定运行后，再逐步尝试引入混合精度训练，以平衡计算效率与精度。
4. **数据加载优化**：为避免在多进程中直接操作NPU张量引发的问题，采用CPU中转数据的策略。训练前将数据置于CPU，训练时再加载到NPU，确保数据加载流程顺畅。 
