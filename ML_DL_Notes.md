# 机器学习、深度学习与代码学习笔记

> 更新规则：后续在本对话中询问机器学习、深度学习或代码知识时，更新此文件。重复询问同一概念时补充原章节，并标记「🔁 需要复习」，记录重复次数。

## PyTorch

### 训练循环
标准顺序：`optimizer.zero_grad()` → 前向计算损失 → `loss.backward()` → `optimizer.step()`。

### 梯度清零
PyTorch 的 `.grad` 默认累加梯度，故通常每轮更新前清零；也可有意累积梯度。

### train() 和 eval()
`model.eval()` 影响 Dropout、BatchNorm 等模块；纯 `nn.Embedding` 一般不受影响。`eval()` 不会关闭梯度，推理可配合 `torch.no_grad()`。

### nn.Module
自定义网络继承 `nn.Module`，在 `__init__` 中定义层，在 `forward` 中描述前向计算。

### Dataset 与 DataLoader
`Dataset` 定义 `__len__` 和 `__getitem__`；`DataLoader` 实现批处理、打乱顺序等。

### 保存和加载模型
`torch.save(model.state_dict(), 'model.pth')` 保存权重；`model.load_state_dict(torch.load('model.pth', weights_only=True))` 恢复。优化器状态需另存。

## 深度学习概念

### Embedding
将离散 ID 映射为稠密向量。`nn.Embedding(100, 32)` 是 100 个 ID 对应的 32 维可学习向量表。

### padding_idx
`nn.Embedding(..., padding_idx=0)` 指定填充项，其嵌入行不会从该层查找操作收到梯度。真实用户或商品 ID 0 不应设为填充。

### Pooling（池化）
将多个值或向量聚合为较少的值或向量，如均值池化和最大池化。含 padding 的序列做均值池化时应使用掩码。

### 自监督学习
利用数据本身构造训练目标，如掩码语言建模、图对比学习。推荐系统可结合 `L = L_BPR + λ L_SSL`。

### One-Hot（独热编码）
向量只有一个元素为 1，其余为 0。one-hot 与嵌入矩阵相乘相当于取出对应行。

### Softmax
`softmax(z_i) = exp(z_i) / sum_j exp(z_j)`，将 logits 转成和为 1 的非负值。`torch.softmax(scores, dim=-1)`；`CrossEntropyLoss` 一般直接接收 logits。

## 推荐系统

### 负采样（Negative Sampling）
从用户未交互商品中抽取训练负样本；未交互不等于不喜欢，因此可能产生假负例。常见策略包括均匀采样、按流行度采样和困难负采样。

```python
import random
all_items = set(range(5))
positive_items = {0, 2}
negative_item = random.choice(list(all_items - positive_items))
```

BPR 损失 `-log(sigmoid(s(u,i) - s(u,j)))` 鼓励正样本商品 `i` 的评分高于负采样商品 `j`。采样方式可能影响非头部商品覆盖。

## 复习状态

以上概念均为首次记录。再次提问时更新原章节、标记「🔁 需要复习」并累计重复提问次数。

## 更新日志

- 2026-10-12：初始化，收录 13 个已问知识点。
