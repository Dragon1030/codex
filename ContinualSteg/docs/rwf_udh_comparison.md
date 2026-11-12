# RwF、Replay/Distillation 与当前 ContinualSteg 的对应关系

## 1. 相关论文到底是哪几篇

### RwF 本身

`Restoring Without Forgetting: Continual Learning Across Image Degradations` 的核心不是 Replay 或 Distillation，而是**冻结共享恢复主干 + 每个退化使用独立 Adapter 路径 + Prototype 路由**。它明确采用无旧样本保存的参数隔离方案。

### Replay Buffer

RwF 的相关工作部分把 Replay 归为持续学习的三类方法之一，并引用：

- Chaudhry et al., **Efficient Lifelong Learning with A-GEM**, 2018：使用少量过去样本约束更新方向。
- Chaudhry et al., **On Tiny Episodic Memories in Continual Learning**, 2019：研究小型 episodic memory 如何保存旧样本。
- Rolnick et al., **Experience Replay for Continual Learning**, 2019：经验回放的经典工作，当前代码的随机抽样 ReplayBuffer 与它的思想相近。

当前 `ReplayBuffer` 只做样本保存、随机抽样和 reservoir sampling；它没有实现 A-GEM 的梯度投影，所以不能称为 A-GEM。

### Distillation

对应论文是：

- Li & Hoiem, **Learning without Forgetting**, ECCV 2016 / TPAMI 2017。

LwF 用旧模型作为 teacher，让新模型在旧任务输出上保持接近。当前代码的形式更接近输出蒸馏：对旧样本分别比较 teacher/student 的 container 和 recovered secret 的 MSE，但它不是原论文的分类 logits 蒸馏。

### Adapter 与 Prompt

- RwF 的 Adapter 设计借鉴了低秩适配思想，论文明确引用 LoRA；每个退化在主干的多个 block 中放置一组 Adapter。
- AIM（ICLR 2023）研究的是冻结图像模型、加入轻量 Adapter 的参数高效迁移。AIM 的 Adapter 放在 Transformer 的 self-attention 后、temporal attention 后以及 MLP 旁路中；AIM 本身不是当前 PromptBank 的来源。
- 当前项目的 `PromptBank` 是任务条件向量的工程实现，属于 task-conditioned prompt/bias，不是 RwF 图中的模块，也不是 AIM 中的核心模块。

## 2. RwF 论文结构图怎么读

论文图从左到右可以分成训练区和推理区。

1. 左侧 `D1, D2, ..., DN` 是按顺序到来的退化域，例如噪声、模糊、雨、雾、低照度。
2. 中间左侧是共享 Encoder Path，蓝色雪花表示主干被冻结；每个 Encoder Level 内有当前域的 Adapter。
3. 中间底部是 Bottleneck/Latent；共享主干负责通用图像恢复特征，Adapter 负责修正当前退化。
4. 中间右侧是 Decoder Path，Decoder 的各个 level 也放置对应域的 Adapter，因此一条路径覆盖整个 Encoder–Decoder，而不是只改一个层。
5. 中央的共享 embedding space 用来生成每个域的 Prototype `μ1 ... μN`。
6. 右侧推理时不提供退化标签：输入先经过浅层 Encoder，得到 embedding；再和所有 Prototype 做 cosine similarity，选择 Adapter id。
7. 选中的 Adapter 在 Encoder 和 Decoder 路径同时激活，其他路径保持不活动，最后得到 restored image。
8. 图中的红色虚线是下采样，绿色虚线是上采样，黄色虚线是 U-Net skip connection；蓝色表示 frozen，橙色表示当前可训练路径。

RwF 的 Adapter 注入位置不是 bottleneck 单点，而是每个主干 block。论文对 Transformer 的表达是：在 attention 后的特征上加入 Adapter，再进入 FFN；对卷积网络则放在 channel-mixing 分支附近。RwF 没有 PromptBank。

## 3. 当前 ContinualSteg 的真实 Adapter/Prompt 位置

当前代码对应 `continual_steg/models.py`。

### HidingNet

```text
cover [B,3,H,W] + secret [B,3,H,W]
        ↓ concat
[B,6,H,W]
        ↓ Conv 6→64, stride=2
[B,64,H/2,W/2]
        ↓ Conv 64→128, stride=2
[B,128,H/4,W/4]
        ↓ Conv 128→256, stride=2
z_h = [B,256,H/8,W/8]
        ↓ SpatialAdapter_t
        ↓ + prompt_bias_t(prompt_t)
        ↓ 3 个 ConvTranspose 上采样层
container [B,3,H,W]
```

### RevealNet

```text
attacked container [B,3,H,W]
        ↓ Conv 3→64, stride=2
[B,64,H/2,W/2]
        ↓ Conv 64→128, stride=2
[B,128,H/4,W/4]
        ↓ Conv 128→256, stride=2
z_r = [B,256,H/8,W/8]
        ↓ SpatialAdapter_t
        ↓ + prompt_bias_t(prompt_t)
        ↓ 3 个 ConvTranspose 上采样层
recovered secret [B,3,H,W]
```

当 `image_size=128` 时，瓶颈空间大小是 `16×16`；当你的训练命令使用 `image_size=64` 时，瓶颈空间大小是 `8×8`。通道数始终是 256。

### Adapter 的精确注入

`SpatialAdapter` 定义为：

```python
x = x + up(GELU(down(GroupNorm(x))))
```

其中：

- `GroupNorm(1, 256)`：对瓶颈特征归一化；
- `down`：`1×1 Conv 256→32`，默认 `adapter_bottleneck=32`；
- `GELU`：非线性变换；
- `up`：`1×1 Conv 32→256`；
- 输出采用残差方式加回原特征。

因此当前项目的 Adapter 是**隐藏端一个 bottleneck Adapter + 恢复端一个 bottleneck Adapter**。五个任务总共会创建 10 个 `SpatialAdapter`，但每个阶段只训练当前 task_id 对应的两组 Adapter。

### Prompt 的精确注入

```python
prompt_t = PromptBank.prompts[task_id]       # [64]
bias_t = prompt_bias[task_id](prompt_t)      # [256]
bias_t = bias_t[:, :, None, None]             # [B,256,1,1]
x = adapter(x) + bias_t                       # broadcast 到 H/8×W/8
```

所以 Prompt 不是额外图片，也不是拼接到输入的通道；它是一个每任务 64 维向量，经线性层变成 256 个通道偏置，在 Adapter 后、Decoder 前注入瓶颈特征。

HidingNet 和 RevealNet 使用同一个 task prompt，但各自有独立的 `prompt_bias` 线性投影。

需要注意：当前 `PromptBank`、`prompt_bias.weight` 和 `prompt_bias.bias` 都是零初始化。这样可以保持旧模型初始行为，但也可能使 prompt 向量和 weight 的梯度长期为零，实际有效的部分更接近每任务 bias。若要验证 Prompt 的独立贡献，应改为非零小幅初始化或加入显式 prompt loss，并重新做 ablation。

## 4. 当前项目与 RwF 的关键差异

| 位置 | RwF | ContinualSteg |
|---|---|---|
| 共享主干 | 从开始冻结的 Restormer/NAFNet | 第一个 clean 阶段训练共享 CNN，后续阶段冻结 |
| Adapter 数量 | 每个 Encoder/Decoder block 一组 | HidingNet 和 RevealNet 各在 bottleneck 一组 |
| Adapter 形式 | LayerNorm + 低秩残差 Adapter | GroupNorm + 1×1 down/up 残差 Adapter |
| Prompt | 没有 PromptBank | 每任务 64 维 Prompt，经 Linear 变成 256 通道 bias |
| 旧数据 | 不保存旧数据 | ReplayBuffer 保存少量旧 cover/secret/task_id |
| Distillation | 不是 RwF 核心机制 | frozen old_model 与当前模型输出 MSE |
| 路由特征 | Encoder Level 1 的均值和标准差 | 当前 `RevealNet.extract_feature()` 的 bottleneck 空间均值 |

因此当前项目是把 RwF 的“路径隔离和路由”思想迁移到图像隐写，而不是 RwF 的逐层 Adapter 原样复现。

## 5. 为什么 clean 阶段与直接 UDH 复现的数值会不同

### 网络容量不同

原始 UDH 代码中 `num_downs=5`：

- `HidingUNet`：5 个下采样 `Conv2d` + 5 个上采样 `ConvTranspose2d`，包含 U-Net skip connection；
- `RevealNet`：6 个同分辨率 `3×3 Conv2d`，没有下采样和上采样；
- 对 `imageSize=128`，HidingUNet 的空间路径是 `128→64→32→16→8→4→8→16→32→64→128`。

当前 ContinualSteg：

- `HidingNet`：3 个下采样卷积 + 3 个上采样反卷积；
- `RevealNet`：3 个下采样卷积 + 3 个上采样反卷积；
- 没有 U-Net 的长跳跃连接；
- 对 `image_size=128`，空间路径是 `128→64→32→16→32→64→128`。

因此当前模型更轻量，但与原始 UDH 不是同一个网络。尤其恢复端原始 UDH 保持高分辨率连续卷积，当前 RevealNet 先压缩再恢复，细节恢复能力不同。

### 输入/输出定义不同

原始 UDH 默认 `cover_dependent=False`，HidingUNet 主要接收 secret，输出一个隐写残差，随后由外部代码计算：

```python
container = cover + hidden_residual
```

当前模型直接把 cover 和 secret 拼接后输入 HidingNet：

```python
container = HidingNet(torch.cat([cover, secret], dim=1))
```

这改变了优化问题：原始 UDH 显式限制隐藏残差，当前模型直接生成 container，只通过 cover loss 约束它接近 cover。

### 损失函数不同

原始 UDH 默认：

```text
loss = MSE(container, cover) + 0.75 × MSE(recovered, secret)
```

当前 clean 阶段没有旧模型，因此蒸馏项为 0，实际为：

```text
loss = L1(container, cover) + L1(recovered, secret)
```

两者在以下方面不同：

- MSE 与 L1 的梯度特性不同；
- secret loss 权重不同，UDH 为 0.75，当前模型默认等权；
- 原始 UDH 使用残差式 container，当前模型直接输出 container；
- 当前模型有 task adapter/prompt，原始 UDH 没有。

### 归一化和优化器不同

原始 UDH 常用 `BatchNorm2d`、Adam、初始学习率 `0.001`；当前模型使用 `GroupNorm`，AdamW，默认学习率 `2e-4`。BatchNorm 的 batch 统计和 GroupNorm 的逐样本统计会产生不同结果。

### 数据配对方式不同

原始 UDH 使用两个独立的 `ImageFolder` 和两个 DataLoader，再通过 `zip` 配对 cover/secret。当前 `PairedImageDataset` 使用一个目录，并通过固定索引公式独立选择 secret。即使图像集合相同，实际配对顺序也不同。

### 指标计算方式不同

原始 UDH 的分析代码按图像、按通道计算 PSNR/SSIM 后再平均；当前 `metrics.py` 的 PSNR 先对整个 batch 的所有像素计算一个 MSE，再转成一个 PSNR，然后对 batch 结果平均。两者不一定相等，尤其不同 batch size 或图像难度差异较大时。

### 训练协议不同

原始 UDH 脚本默认约 65 个 epoch、每 epoch 最多 2000 个 iteration；当前项目是每个持续学习任务 50 个 epoch，并且 clean 阶段之后还要继续经过量化、高斯、resize、crop 任务。若比较的是最终 checkpoint 而不是 clean 阶段 checkpoint，结果也不能直接与 UDH clean baseline 比较。

## 6. 如何做公平的 clean 对照

要判断差异到底来自网络还是训练协议，建议固定以下条件：

1. 同一 DIV2K train/val 文件列表和同一 cover-secret 配对；
2. 同一 `image_size`，建议先统一为 128；
3. 同一 batch size、epoch/iteration 数和随机种子；
4. 同一损失：都使用 MSE，或都使用 L1；
5. 同一 container 定义：都采用 `cover + residual`，或都直接输出 container；
6. 同一归一化：都使用 BatchNorm 或都使用 GroupNorm；
7. 同一 PSNR/SSIM 实现，统一为逐图像计算后再平均；
8. 只比较 clean 阶段，不把后续持续任务的 checkpoint 当作 clean baseline；
9. 额外记录 HidingNet、RevealNet、Adapter 和 Prompt 的参数量。

最有解释力的四组实验是：

```text
A. 原始 UDH + 原始损失 + 原始指标
B. 当前 CNN + 原始损失 + 原始指标
C. 原始 UDH + 当前损失 + 当前指标
D. 当前 CNN + 当前损失 + 当前指标
```

A 与 B 主要反映网络结构差异，B 与 D 主要反映损失/评估协议差异。这样才能说明 clean 阶段指标变化究竟来自模型容量、数据、损失还是评价方式。
