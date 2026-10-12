# 机器学习、深度学习与代码学习笔记

> 更新规则：本对话中出现的新概念加入对应章节；重复问题在原条目补充，不重复建条目，并标注 **🔁 不熟悉／需要复习**。示例代码优先采用可运行的 PyTorch 代码。记录日期：2026-10-12。

### 维护口径

- 更新前先全文检索，按“同一核心概念”判断是否重复，而不是只比较问题文字。
- 新概念：写入对应章节，并同步更新“学习索引”和“后续更新记录”。
- 重复概念：在原条目继续补充，不另建同义条目；标记 `🔁 不熟悉／需要复习`，并记录“提问次数”和“最近提问日期”。
- 只有同一概念从第 2 次提问开始才标记为不熟悉；相关但实质不同的概念不误判为重复。
- 代码类结论优先保留可运行示例、关键输入输出和适用边界；尚未验证的判断明确写成假设。

## 学习索引

| 主题 | 核心内容 | 状态 |
|---|---|---|
| PyTorch 训练循环 | 清零、前向与损失、反向、更新 | 已记录 |
| 梯度 `.grad` | 默认累加，通常每轮清零 | 已记录 |
| `model.eval()` | 评估模式不等于关闭梯度 | 已记录 |
| `nn.Embedding` | 按 ID 查找可训练向量 | 已记录 |
| `padding_idx` | 填充索引及其梯度行为 | 已记录 |
| Pooling | 对特征或向量进行聚合 | 已记录 |
| Dataset / DataLoader | 样本接口与批量读取 | 已记录 |
| `nn.Module` | 自定义模型与 `forward` | 已记录 |
| 保存与恢复模型 | `state_dict` / `load_state_dict` | 已记录 |
| 自监督学习 | 从数据本身构造学习目标 | 已记录 |
| One-Hot | 用单个 1 表示类别 | 已记录 |
| Softmax | 将分数转为和为 1 的分布 | 已记录 |
| 负采样中的噪声词 | 从噪声分布采样、充当负样本的词 | 初次记录 |

> **复习标记说明**：目前尚未发现本笔记中同一概念被明确重复提问的情况。后续重复提问时，将在原条目加入 `🔁 不熟悉／需要复习`，并记录重复提问次数和新补充的理解。

## 1. PyTorch 基础

### 1.1 标准训练循环

四步：① `optimizer.zero_grad()` 清零梯度；② 前向传播并计算 `loss`；③ `loss.backward()` 计算梯度；④ `optimizer.step()` 更新参数。

```python
for x, y in loader:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

### 1.2 为什么要清零 `.grad`？

PyTorch 默认**累加**梯度，反复调用 `backward()` 会将新梯度加到已有 `.grad`。通常每次更新前清零，避免意外混合多个 batch；有意使用梯度累积时则不必每个 batch 清零。

### 1.3 `model.train()`、`model.eval()` 和 `torch.no_grad()`

- `train()` / `eval()` 控制部分模块的行为，如 Dropout、BatchNorm。
- 纯 `nn.Embedding` 在相同权重、输入下，切换模式通常不会改变输出。
- `eval()` **不会**自动禁用梯度；推理时通常搭配 `torch.no_grad()` 或 `torch.inference_mode()`。

### 1.4 `nn.Module`、Dataset 与 DataLoader

- `nn.Module`：定义模型结构；`forward()` 定义前向计算。
- `Dataset.__len__()`：返回样本数量；`Dataset.__getitem__(idx)`：返回单个样本。
- `DataLoader`：按 batch 取数据，可用 `shuffle=True` 每轮打乱顺序。

### 1.5 保存和恢复参数

```python
torch.save(model.state_dict(), "model.pth")
restored = MyModel()
restored.load_state_dict(torch.load("model.pth", weights_only=True))
restored.eval()
```

`state_dict()` 保存参数和持久化缓冲区，不保存 Python 模型类定义。若需精确恢复训练过程，通常还要保存优化器状态、epoch 等。

### 1.6 一个可运行的最小训练例子

```python
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(0, 1, 100).unsqueeze(1)
        self.y = 2 * self.x + 1
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    def forward(self, x):
        return self.linear(x)

loader = DataLoader(MyDataset(), batch_size=10, shuffle=True)
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
for epoch in range(200):
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), "model.pth")
restored = MyModel()
restored.load_state_dict(torch.load("model.pth", weights_only=True))
```

模型学习 `y = 2x + 1`；初始参数随机，具体收敛速度可能变化。原始示例采用 `x=0..99` 和小学习率，训练稳定性较差，这里改为归一化范围。

## 2. 向量与表示学习

### 2.1 Embedding（嵌入向量）

将用户、商品、词等对象表示为向量。`nn.Embedding(100, 32)` 创建 100 行、每行 32 维的可训练查找表。输入 ID 后返回相应行；语义相似性是否学得好取决于训练目标和数据。

### 2.2 `padding_idx`

`nn.Embedding(5, 3, padding_idx=0)` 将 ID 0 作为填充索引，默认其向量初始化为 0，正常通过该 Embedding 层反向传播时不更新该行。它**不自动排除**填充项对均值池化等后续计算的影响；仍需 mask。普通 LightGCN 用户/商品 ID 0 是真实节点时不应设为 padding。

### 2.3 Pooling（池化）

把多个向量或特征聚合：平均池化逐维取平均，最大池化逐维取最大。例：`[[1,2],[3,4]]` 的 mean pooling 为 `[2,3]`。

```python
x = torch.tensor([[1., 2.], [3., 4.]])
print(x.mean(dim=0))       # tensor([2., 3.])
print(x.max(dim=0).values) # tensor([3., 4.])
```

文本向量的平均池化通常需要 mask 排除 padding；LightGCN 常将各传播层节点向量平均聚合。

### 2.4 One-Hot（独热向量）

有 K 个类别时，使用 K 维向量，其中恰有一个位置为 1，其余为 0。例如类别 1（共 3 类）是 `[0,1,0]`。它表示身份，不直接表达类别相似性。Embedding 查表可在数学上视为 one-hot 向量乘以 Embedding 权重矩阵，但无需显式构造 one-hot。

```python
import torch.nn.functional as F
print(F.one_hot(torch.tensor([0, 1, 2]), num_classes=3))
```

## 3. 学习范式和函数

### 3.1 自监督学习（Self-Supervised Learning）

无需人工逐项标注，而从数据本身构造训练目标。例如遮住句子中的词并预测被遮住的词；图对比学习将同一节点在不同扰动视图下的表示拉近。自监督学习有训练目标，不等于完全没有标签信号。

推荐系统中可用 `L = L_BPR + λ * L_SSL` 联合训练，但 SSL **不保证**改善长尾覆盖，需要实验验证。

### 3.2 Softmax

对分数 `z`：

\[
\operatorname{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
\]

输出非负、总和为 1，保持分数排序。例：`softmax([2,1,0]) ≈ [0.665,0.245,0.090]`。

```python
scores = torch.tensor([2., 1., 0.])
print(torch.softmax(scores, dim=0))
```

与 Sigmoid 不同：Sigmoid 逐元素独立转换，不要求总和为 1。LightGCN+BPR 常使用 `sigmoid(s_pos - s_neg)`，通常不需要 Softmax；Transformer 注意力会使用 Softmax。`nn.CrossEntropyLoss` 接收原始 logits，不应预先调用 Softmax。

## 4. 文本数据处理

### 4.1 负采样中的噪声词（Noise Words）

在 word2vec 的负采样中，噪声词是：**从预定义的噪声分布中抽取、用来充当负样本的词**。模型需要学习：真实上下文词与中心词应该匹配，而这些采样词不应该与中心词匹配。

例如句子是“我 喜欢 这个 游戏”，中心词是“喜欢”，真实上下文词“我”是正样本。若从词表中抽到“飞机”“香蕉”，它们在这次训练中就是噪声词（负样本）。训练目标是提高“喜欢—我”的匹配分数，降低“喜欢—飞机”和“喜欢—香蕉”的匹配分数。

这里的“噪声”不是说“飞机”或“香蕉”本身是乱码或无意义词，也不是说要把它们从数据中删除。它只表示：**相对于当前中心词和上下文窗口，它们被采样为不匹配的对照项**。这样每一步只需计算少量负样本，而不必遍历整个词表。

注意：在其他文本清洗语境中，“噪声词”有时也泛指乱码、广告模板等干扰文本；但《动手学深度学习》“近似训练”这一节采用的是上面的负采样含义。

来源：[《动手学深度学习》14.2 节“近似训练”](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html)

- 提问次数：1
- 最近提问日期：2026-10-12
- 状态：首次记录，暂不标记为不熟悉

## 5. 后续更新记录

| 日期 | 变更 | 复习状态 |
|---|---|---|
| 2026-10-12 | 根据本对话已有问题建立初始笔记（12 个主题） | 暂无明确重复提问 |
| 2026-10-12 | 固定自动更新、概念级去重、重复次数与复习标记规则 | 规则已建立 |
| 2026-10-12 | 新增“负采样中的噪声词”，说明其作为负样本的作用 | 首次提问 |
