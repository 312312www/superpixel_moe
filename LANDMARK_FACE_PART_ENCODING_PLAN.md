# Landmark 人脸部位编码方案

## 1. 目标

Landmark 模块的目标不是把全部人脸关键点坐标直接输入模型，而是：

> 利用现成的人脸关键点检测器判断每个 superpixel 属于哪些人脸部位，再将部位语义编码加入对应的 Region Token。

推荐使用 Google **MediaPipe Face Landmarker**。该方案不需要自行训练关键点检测器，并能直接输出稠密人脸关键点。

- 官方接口文档：<https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceLandmarker>

Landmark 分支只负责产生人脸部位语义，不参与 Backbone 特征提取。

## 2. 整体流程

```text
256×256 对齐人脸
       │
       ├── SLIC
       │     ↓
       │  Fine / Medium / Coarse masks
       │
       └── MediaPipe Face Landmarker
             ↓
          人脸关键点
             ↓
          人脸部位 masks
             ↓
计算每个 superpixel 与各部位的重叠比例
             ↓
得到每个 Region 的部位概率分布
             ↓
生成 Landmark 人脸部位编码
             ↓
Region Token + 几何编码 + 部位编码
             ↓
          LayerNorm
             ↓
             MoE
```

## 3. 人脸部位定义

第一版不为数百个 Landmark 点分别编码，而是将关键点归纳为 11 类稳定的人脸部位：

| ID | 英文名称 | 中文含义 |
|---:|---|---|
| 0 | `unknown` | 未知区域或背景 |
| 1 | `left_eyebrow` | 左眉 |
| 2 | `right_eyebrow` | 右眉 |
| 3 | `left_eye` | 左眼 |
| 4 | `right_eye` | 右眼 |
| 5 | `nose` | 鼻部 |
| 6 | `mouth` | 嘴部 |
| 7 | `left_cheek` | 左脸颊 |
| 8 | `right_cheek` | 右脸颊 |
| 9 | `forehead` | 额头 |
| 10 | `chin` | 下巴 |

这样设计的原因：

- 同一解剖部位附近的多个 Region 可以共享部位语义；
- 不同图片的 superpixel 边界会变化，但人脸部位类别相对稳定；
- 符合“同一人脸部位允许重复编码”的设计要求；
- 比给每个关键点设置独立 ID 更容易训练和解释。

后续可根据实验结果进一步拆分上唇、下唇、鼻梁、鼻尖、内外眼角或脸颊子区域，但第一版不建议过度细分。

## 4. Landmark 检测模块

新增文件：

```text
fas_moe/landmarks.py
```

建议接口：

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class FaceLandmarkResult:
    points: np.ndarray       # [N, 2] 或 [N, 3]，归一化坐标
    detected: bool
    confidence: float
    face_bbox: np.ndarray    # [x1, y1, x2, y2]
    source: str              # mediapipe / fallback
    reason: str              # 成功或失败原因


class MediaPipeFaceLandmarker:
    def detect(self, image: np.ndarray) -> FaceLandmarkResult:
        ...
```

建议配置：

```text
num_faces = 1
min_face_detection_confidence = 0.5
min_face_presence_confidence = 0.5
```

若输入已经是裁剪并对齐后的人脸，只保留置信度最高或面积最大的人脸。

## 5. 从关键点构造部位 Mask

新增文件：

```text
fas_moe/face_parts.py
```

输出格式：

```python
part_masks.shape == [P, H, W]
```

其中：

- `P = 11`：人脸部位数量；
- `H`、`W`：输入图像高度和宽度。

### 5.1 直接由轮廓点构造的部位

根据 MediaPipe Landmark 索引取得轮廓点并填充多边形：

- 左眼；
- 右眼；
- 左眉；
- 右眉；
- 鼻部；
- 嘴部；
- 人脸轮廓。

### 5.2 需要组合推导的部位

脸颊、额头和下巴没有完全独立的封闭轮廓，需要在人脸椭圆区域内根据关键点和几何位置构造：

- 左脸颊：左眼下方、鼻子左侧、人脸轮廓内部；
- 右脸颊：右眼下方、鼻子右侧、人脸轮廓内部；
- 额头：眉毛上方、人脸轮廓内部；
- 下巴：嘴部下方、人脸轮廓内部。

构造顺序：

1. 生成完整的 face oval mask；
2. 划分额头、脸颊和下巴等大区域；
3. 用眼睛、眉毛、鼻子和嘴部等局部多边形覆盖对应区域；
4. 人脸轮廓外及未可靠归属的位置设置为 `unknown`。

必须保存可视化结果，确认左右部位方向和多边形边界正确。

## 6. 将 Superpixel 映射到人脸部位

对于第 `k` 个 superpixel 区域 `Sₖ`，计算它与第 `p` 个人脸部位 Mask `Mₚ` 的重叠比例：

$$
r_{k,p}=\frac{|S_k\cap M_p|}{|S_k|}
$$

每个 Region 得到一个部位概率分布：

$$
r_k=[r_{k,0},r_{k,1},\ldots,r_{k,P-1}]
$$

要求：

```text
part_distribution.shape = [K, 11]
每行元素非负
每行概率和约等于 1
```

### 6.1 硬部位 ID

选择重叠率最大的部位：

$$
p_k=\arg\max_p r_{k,p}
$$

如果最大重叠率低于阈值，例如 `0.3`，则使用 `unknown`：

```python
part_id = overlap.argmax()
if overlap.max() < 0.3:
    part_id = UNKNOWN
```

该方法实现简单，但一个 Region 横跨多个部位时会丢失信息，并且容易因边界变化产生标签跳变。

### 6.2 软部位分布（推荐）

保留完整的重叠概率，例如：

```text
Region 7:
left_eye   = 0.62
left_cheek = 0.31
unknown    = 0.07
```

软分布更适合不规则 superpixel，也是第一版正式实现的推荐方式。

## 7. 部位编码

在模型中增加可学习部位 Embedding：

```python
self.part_embedding = nn.Embedding(
    num_face_parts,
    feature_channels,
)
```

### 7.1 硬编码

$$
e_k^{part}=E[p_k]
$$

```python
part_encoding = self.part_embedding(part_ids)
```

### 7.2 软编码（推荐）

$$
e_k^{part}=\sum_p r_{k,p}E[p]
$$

```python
part_encoding = part_distribution @ self.part_embedding.weight
```

若：

```text
part_distribution: [B, K, P]
embedding weights:  [P, C]
```

则输出：

```text
part_encoding: [B, K, C]
```

## 8. 与 Region Token 融合

区域平均池化得到 Region Token：

$$
z_k=\operatorname{AveragePool}(F,S_k)
$$

现有五维几何位置编码为：

$$
e_k^{geo}=\operatorname{MLP}(x_k,y_k,a_k,w_k,h_k)
$$

加入部位编码后：

$$
\tilde z_k=
z_k+e_k^{geo}+e_k^{part}+e^{scale}
$$

其中 `eˢᶜᵃˡᵉ` 是可选的尺度编码，用于区分 Fine、Medium 和 Coarse。

推荐实现：

```python
pooled = pool_regions(feature_map, labels)
geometry_encoding = self.geometry_encoder(positions)
part_encoding = part_distribution @ self.part_embedding.weight
scale_encoding = self.scale_embedding(scale_id)

tokens = (
    pooled
    + geometry_encoding
    + part_encoding
    + scale_encoding
)

tokens = self.token_norm(tokens)
```

相加后使用 `LayerNorm`，避免不同编码的数值范围不一致。

第一版推荐使用“相加 + LayerNorm”。后续可通过消融实验比较拼接后线性投影：

```python
combined = torch.cat(
    [pooled, geometry_encoding, part_encoding],
    dim=-1,
)
tokens = self.token_projection(combined)
```

## 9. 编码权重与归一化

如果不同编码对训练影响差异过大，可以增加可学习权重：

$$
\tilde z_k=
z_k+
\lambda_{geo}e_k^{geo}+
\lambda_{part}e_k^{part}+
\lambda_{scale}e^{scale}
$$

```python
self.geometry_weight = nn.Parameter(torch.tensor(1.0))
self.part_weight = nn.Parameter(torch.tensor(1.0))
self.scale_weight = nn.Parameter(torch.tensor(1.0))
```

该设计作为后续扩展，第一版可以先使用等权相加和 `LayerNorm`。

## 10. Landmark 检测失败处理

正式训练中可能遇到侧脸、模糊、遮挡、低分辨率、打印攻击、屏幕攻击或面具等情况。Landmark 失败不能导致训练或推理中断。

### 10.1 正常检测

使用计算得到的部位软分布。

### 10.2 检测失败

所有 Region 回退到 `unknown`：

```python
part_distribution = np.zeros((region_count, num_face_parts), dtype=np.float32)
part_distribution[:, UNKNOWN] = 1.0
```

此时模型仍可使用：

- Backbone 特征；
- Region Pooling；
- 几何位置编码；
- MoE 分类分支。

### 10.3 低置信度

可以根据质量分数 `q` 减弱部位编码：

$$
e_k^{part}=
q\sum_p r_{k,p}E[p]+(1-q)E[\mathrm{unknown}]
$$

第一版可先只实现成功/失败回退；置信度门控作为后续扩展。

## 11. 预计算与缓存

不要在每个 epoch 中重复运行 MediaPipe。建议在预处理阶段缓存：

```text
cache/<sample_id>/
  landmarks.npy
  landmark_meta.json
  part_distribution_128.npy
  part_distribution_064.npy
  part_distribution_016.npy
```

推荐至少保存：

```text
landmarks.npy               [N, 2] 或 [N, 3]
part_distribution_128.npy   [128, 11]
part_distribution_064.npy   [64, 11]
part_distribution_016.npy   [16, 11]
```

如果训练只使用亮度、颜色、模糊和压缩等非几何增强，缓存可以直接复用。

如果使用随机裁剪、旋转、水平翻转或仿射变换，则必须同步变换：

- 图像；
- Landmark；
- superpixel mask；
- 人脸部位 mask。

第一版建议暂不使用会改变空间坐标的增强。

## 12. 水平翻转处理

水平翻转时，Landmark 横坐标变换为：

$$
x'=1-x
$$

同时必须交换左右部位标签：

```text
left_eye     ↔ right_eye
left_eyebrow ↔ right_eyebrow
left_cheek   ↔ right_cheek
```

否则模型会收到错误的人脸解剖语义。

## 13. 可视化与质量检查

每张测试图建议输出：

1. 原图及 Landmark 点；
2. 人脸部位彩色 Mask；
3. 三尺度 superpixel 边界；
4. 每个 superpixel 对应的主要部位颜色；
5. 必要时显示一个 Region 的完整部位软分布。

重点检查：

- 左右眼、眉毛和脸颊是否颠倒；
- 嘴部、鼻部和眼部多边形是否闭合；
- 脸颊、额头和下巴覆盖是否合理；
- 人脸轮廓外是否保持为 `unknown`；
- Fine、Medium、Coarse 三个尺度是否保持一致的部位语义。

## 14. 文件改造清单

### 14.1 新增文件

```text
fas_moe/landmarks.py
fas_moe/face_parts.py
```

### 14.2 修改 `fas_moe/segmentation.py`

在 `SuperpixelViews` 中增加：

```python
part_distributions: Dict[int, np.ndarray]
```

在元数据中记录：

```text
landmarks_enabled
landmarks_detected
landmark_reason
landmark_cache_hit
```

### 14.3 修改 `fas_moe/model.py`

增加：

```python
part_embedding
token_norm
```

可选增加：

```python
scale_embedding
geometry_weight
part_weight
scale_weight
```

### 14.4 修改 `run_moe.py`

导出：

```text
part_distribution_128.npy
part_distribution_064.npy
part_distribution_016.npy
summary.json
```

建议进一步增加：

```text
landmark_visualization.png
part_visualization.png
```

### 14.5 修改 `requirements.txt`

加入与当前 Python 环境兼容的 MediaPipe 版本，并提供 Face Landmarker 模型的官方下载说明。

## 15. 测试要求

### 15.1 Landmark 检测测试

- 输出点数符合模型预期；
- 坐标有限并处于合理范围；
- 无人脸或模型缺失时返回失败状态，而不是抛出未处理异常。

### 15.2 部位 Mask 测试

- Mask 形状为 `[11, H, W]`；
- 部位 ID 合法；
- 人脸轮廓外为 `unknown`；
- 左右部位没有交换；
- 不包含 NaN 或无穷值。

### 15.3 Region 映射测试

- 每个尺度输出 `[K, 11]`；
- 每行概率和约等于 1；
- 不出现 NaN；
- 检测失败时每行均为 `unknown=1`。

### 15.4 模型测试

- 加入部位编码后 token 维度保持 `[B, K, 512]`；
- 前向和反向传播成功；
- `part_embedding` 能获得有限梯度；
- `unknown` 输入不会导致模型报错。

## 16. 消融实验

至少比较以下设置：

| 实验 | 几何编码 | Landmark 部位编码 | 目的 |
|---|---:|---:|---|
| A | 否 | 否 | 无位置语义基线 |
| B | 是 | 否 | 验证几何位置编码 |
| C | 否 | 是 | 单独验证部位语义 |
| D | 是 | 是 | 完整位置编码方案 |
| E | 是 | 随机打乱 | 排除参数量或随机标签带来的假提升 |

实验 E 很重要。如果随机部位编码与真实 Landmark 的性能接近，说明模型可能没有真正利用人脸解剖语义。

同时统计：

- Landmark 检测成功率；
- `unknown` Region 比例；
- 不同姿态和遮挡条件下的检测成功率；
- 不同攻击类型上的性能；
- 跨数据集 ACER、HTER 和 AUC。

## 17. 第一版推荐实现

推荐采用以下最小完整流程：

```text
MediaPipe Face Landmarker
→ 11类人脸部位 Mask
→ 计算 superpixel 与各部位的重叠比例
→ 保留软部位分布
→ 加权查询可学习 Part Embedding
→ Region Token + Geometry Encoding + Part Encoding
→ LayerNorm
→ MoE
```

最终 Region Token 表达为：

$$
\boxed{
\tilde z_k=
\operatorname{LayerNorm}
\left(
z_k+e_k^{geo}+\sum_p r_{k,p}E[p]
\right)
}
$$

该方案使用现成 Landmark 模型，符合“同一人脸解剖部位可以共享和重复编码”的设计要求，同时能够处理一个不规则 Region 横跨多个部位的情况。

## 18. 第一版验收标准

实现完成后应满足：

- MediaPipe 能对真实人脸输出 Landmark；
- 三个尺度分别输出 `[128,11]`、`[64,11]` 和 `[16,11]` 的软分布；
- 每行概率和约等于 1；
- Landmark 失败时自动回退到 `unknown`；
- `part_encoding` 实际加入 MoE 前的 Region Token；
- `part_embedding` 能参与反向传播；
- 单元测试、真实图片前向和最小训练步骤全部通过。
