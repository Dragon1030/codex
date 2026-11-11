# ContinualSteg 当前模型结构与训练流程

这张图对应仓库当前实际代码（`train.py`、`continual_steg/models.py`、`tasks.py`），不是原始 HiNet 或原始 UDH 的网络结构替换版。当前实现是一个紧凑的 UDH 风格 CNN，并在隐藏端和恢复端加入任务 Adapter、Prompt 和 Replay/Teacher 接口。

更适合汇报和快速理解的简化图：
[`continual_steg_structure_clean.svg`](continual_steg_structure_clean.svg) ·
[`continual_steg_structure_clean.png`](continual_steg_structure_clean.png)

## 1. 总体流程

```mermaid
flowchart TD
    A[DIV2K train / val image folders] --> B[PairedImageDataset]
    B --> B1[Read two images independently\ncover and secret]
    B1 --> B2[Resize to 128x128\nToTensor -> [B,3,128,128]]
    B2 --> C[Task sequence\nclean -> quantized -> Gaussian -> resize -> crop]

    C --> D[For each task: 50 epochs]
    D --> E[Concatenate cover + secret\n[B,6,128,128]]
    E --> F[HidingNet]
    F --> G[Container image\n[B,3,128,128]]
    G --> H[Task attack / degradation]
    H --> I[Attacked container]
    I --> J[RevealNet]
    J --> K[Recovered secret\n[B,3,128,128]]

    G --> L[Cover loss: L1(container, cover)]
    K --> M[Secret loss: L1(recovered, secret)]
    R[ReplayBuffer old cover/secret/task_id] --> N[Old-task teacher comparison]
    N --> O[Distillation loss]
    L --> P[Total loss]
    M --> P
    O --> P
    P --> Q[Backpropagation + AdamW]

    Q --> S[End of current task]
    S --> T[Update task prototype\nreveal feature mean + normalize]
    S --> U[Add current batch to replay buffer\nreservoir sampling]
    S --> V[Evaluate all seen tasks]
    V --> W[metrics.csv + stage checkpoint]

    X[Inference: container + known task_id] --> J
    Y[Inference: unknown task] --> Z[Prototype cosine routing]
    Z --> J
```

## 2. 模型内部结构

```mermaid
flowchart LR
    subgraph H[HidingNet: 载体生成]
        C[Cover\n[B,3,128,128]] --> CAT[Concat with secret\n[B,6,128,128]]
        S[Secret\n[B,3,128,128]] --> CAT
        CAT --> E1[Conv 6->64, stride 2\n[B,64,64,64]]
        E1 --> E2[Conv 64->128, stride 2\n[B,128,32,32]]
        E2 --> E3[Conv 128->256, stride 2\n[B,256,16,16]]
        E3 --> HA[Task Adapter + Prompt bias]
        HA --> HD[3 ConvTranspose blocks\nupsample x8]
        HD --> CONT[Container\n[B,3,128,128]]
    end

    CONT --> ATT[Attack selected by task]

    subgraph R[RevealNet: 秘密恢复]
        ATT --> R1[Conv 3->64, stride 2\n[B,64,64,64]]
        R1 --> R2[Conv 64->128, stride 2\n[B,128,32,32]]
        R2 --> R3[Conv 128->256, stride 2\n[B,256,16,16]]
        R3 --> RA[Task Adapter + Prompt bias]
        RA --> RD[3 ConvTranspose blocks\nupsample x8]
        RD --> REC[Recovered secret\n[B,3,128,128]]
    end

    PB[PromptBank\none prompt per task] --> HA
    PB --> RA
    PR[Prototype bank\none 256-D vector per task] --> ROUTE[Cosine route for unknown task]
    ROUTE --> RA
```

## 3. 每个步骤对应什么

| 步骤 | 当前代码中的实现 | 输入 | 输出 / 作用 |
|---|---|---|---|
| 1. 读取数据 | `PairedImageDataset` | `train_dir` 或 `val_dir` | 从同一目录独立取 cover 和 secret；不是固定成对文件 |
| 2. 预处理 | `Resize((128,128))` + `ToTensor()` | PIL RGB 图像 | `[B,3,128,128]`，像素范围约为 `[0,1]` |
| 3. 选择任务 | `get_tasks()` + `apply_attack()` | 当前 `task_id` | 任务顺序为 clean、quantized、Gaussian、resize、crop |
| 4. 生成载体 | `HidingNet.forward()` | cover、secret、task prompt | container；先把两张图拼成 6 通道，再编码、适配、解码 |
| 5. 模拟退化 | `apply_attack(container, task)` | container | 得到训练/评估时的 attacked container；quantized 是 `50/255` 量化，不是真实 JPEG |
| 6. 恢复秘密 | `RevealNet.forward()` | attacked container、task prompt | recovered secret |
| 7. 当前任务损失 | `train_one_task()` | container/recovered 与目标图 | `cover L1 + secret L1` |
| 8. 旧任务约束 | `ReplayBuffer` + `old_model` | 旧 cover、secret、task_id | teacher/student 的 container 与 recovered MSE；当前实现需注意其梯度连接限制 |
| 9. 参数更新 | `AdamW` | total loss | 更新当前允许训练的参数 |
| 10. 任务隔离 | `train_only_task(task_id)` | 当前 task_id | 后续阶段冻结共享主干和旧任务 Adapter，只训练当前 Adapter、Prompt bias、Prompt |
| 11. 记忆更新 | `ReplayBuffer.add()` | 当前任务第一批 cover/secret | 以 reservoir sampling 保留最多 `replay_capacity` 个样本 |
| 12. 路由原型 | `update_prototype()` | 当前任务 container | 从 RevealNet 中间特征得到 256-D 归一化 prototype |
| 13. 评估与保存 | `evaluate()` + `save_stage()` | 所有已见任务 | 写入 `metrics.csv`，保存 `stage_XX.pt` 和 `latest.pt` |

## 4. 持续学习具体体现在哪里

1. **任务按顺序到来**：模型不是一次性同时训练五种退化，而是依次处理 `clean → quantized → Gaussian → resize → crop`。
2. **每个任务有独立参数**：HidingNet 和 RevealNet 各自有一个 `SpatialAdapter`、一个 `prompt_bias`；PromptBank 也为每个任务保存一个 prompt。
3. **旧参数被冻结**：进入新任务后，`train_only_task()` 只打开当前任务的 Adapter 和 Prompt，旧任务路径保持不变。
4. **旧样本保留**：ReplayBuffer 保存少量旧任务的 cover、secret 和 task_id，用于后续阶段比较或重放。
5. **未知任务可路由**：如果推理时没有 task_id，`route()` 将输入 container 的 256-D 特征与各任务 prototype 做余弦相似度，选择最接近的任务路径。
6. **遗忘被显式测量**：每个阶段都重新评估所有已见任务，比较旧任务的最终 secret PSNR 与其历史最好值，得到 forgetting/BWT。

## 5. 当前实现边界

- 这是当前仓库的**紧凑 UDH 风格 CNN 原型**；`integrations/hinet_patch.md` 和 `integrations/udh_patch.md` 是接入说明，当前模型文件尚未替换为原始 HiNet 可逆网络或原始 UDH 网络。
- 代码中的 `jpeg` 任务是可微的像素量化近似，不代表真实 JPEG 编解码。
- 当前训练代码的 distillation 分支主要比较旧任务路径；由于旧任务 Adapter 在持续阶段被冻结，不能仅凭现有 `full/no_distill` 结果声称 distillation 已被单独证明有效。
- 当前 route 使用已知任务 prototype；未知退化检测和路由准确率尚未单独评估。
