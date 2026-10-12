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

### 负采样（Negative Sampling） 🔁 需要复习
**提问记录**：2026-10-12 首次提问；2026-10-12 再次提问。**重复提问次数：1**。

**一句话理解**：推荐系统知道用户买过什么，但通常不知道用户真正不喜欢什么；训练时从用户**未交互**的商品中抽取一部分作为“暂定负样本”，这就叫负采样。

**例子**：用户 A 买过手机、耳机（正样本）。候选商品有电脑、口红、香水，随机抽中口红（负样本），形成训练三元组 `(用户A, 手机, 口红)`。模型学习让手机的推荐得分高于口红。

**注意**：未交互 ≠ 不喜欢。用户可能没看见口红，因此可能抽到**假负例（false negative）**。

```python
import random
all_items = set(range(5))  # 所有商品 ID：0,1,2,3,4
positive_items = {0, 2}    # 用户购买过的商品
candidates = list(all_items - positive_items)  # [1,3,4]（顺序不保证）
negative_item = random.choice(candidates)
print(negative_item)
```

**和 BPR Loss 的关系**：负采样选择负商品 `j`，BPR 使用 `(u, i, j)` 优化排序：
`L_BPR = -log(sigmoid(s(u,i) - s(u,j)))`，其中 `i` 为正商品。目标是 `s(u,i) > s(u,j)`。

**常见策略**：
- 均匀负采样：所有未交互商品等概率抽取，简单、适合作为基线。
- 按流行度采样：按照商品热度设置概率，可能改变热门/长尾商品的训练权重。
- 困难负采样：挑模型目前容易误判为正样本的商品，训练信号更强，但假负例风险可能增加。

**与你的 LightGCN + BPR 研究的关系**：先固定均匀负采样作为基线，记录随机种子、候选集合和每个正样本的负样本数。改变采样策略可能影响准确率及非头部商品覆盖率，因此比较实验要控制变量。

**易混淆点**：负采样不是“找出用户明确讨厌的商品”，而是从缺乏正向交互记录的商品中抽样。
## 复习状态

负采样：🔁 需要复习（重复提问 1 次）。其余概念暂未标记。再次提问时继续补充原章节并累计次数。

## 更新日志

- 2026-10-12：初始化，收录 13 个已问知识点。
- 2026-10-12：负采样再次提问，补充例子、BPR 关系和常见策略，标记 🔁 需要复习（重复 1 次）。
