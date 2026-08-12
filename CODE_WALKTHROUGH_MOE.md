# 新版代码逐文件说明

本文按“输入如何流动”解释新版最简路径。行号以当前文件为准；后续修改代码后，以函数名和相邻注释为准，不要依赖旧行号。

## 1. `fas_moe/segmentation.py`

- `SuperpixelConfig`：保存图像尺寸、三层区域数、SLIC 紧致度、平滑系数和迭代次数。
- `SuperpixelViews`：统一保存一张图像的 RGB、标签、19 维区域特征、邻接边和几何位置。
- `_segment_level`：对一个目标区域数运行标准 SLIC。如果连通修正后区域太少，逐次增加 SLIC 请求数；如果区域太多，则交给 `_merge_to_target`。
- `_merge_to_target`：只在共享边界的区域之间合并，并使用 Lab 颜色距离和面积差作为确定性代价。它的职责只有“把数量修正到目标值”，不建立跨尺度父子层级。
- `_geometry_features`：为每个区域计算中心坐标、面积比例和包围盒宽高，作为位置编码输入。
- `segment_views`：依次生成 128、64、16 三个视图，并调用 `features.py` 的 `extract_region_features` 生成 19 维手工特征。

## 2. `fas_moe/io.py` 与 `fas_moe/features.py`

- `io.py/load_input`：读取 JPG/PNG/BMP/TIFF 或 NPY 中指定样本。
- `io.py/prepare_image`：识别 HWC/CHW 与常见数值范围，统一成 `256x256` RGB `uint8`。
- `features.py/extract_region_features`：计算每个区域的 RGB、Lab、梯度、面积、中心、周长和紧致度，共 19 维。

## 2.1 `fas_moe/landmarks.py` 与 `fas_moe/face_parts.py`

- `landmarks.py/detect_face_landmarks`：调用 Google MediaPipe Face Landmarker；模型缺失、依赖缺失或检测失败时返回失败状态而不抛出到训练循环。
- `face_parts.py/landmarks_to_part_masks`：把 478 点网格归纳为 11 类互斥部位，未被脸部覆盖的像素属于 `unknown`。
- `face_parts.py/part_distribution_for_labels`：计算每个 superpixel 与各部位的软重叠分布 `[K,11]`，并保证每行和为 1。
- `segmentation.py` 使用图像内容、模型标识和分割配置构造缓存键，避免每个 epoch 重复运行 MediaPipe。

## 3. `fas_moe/backbone.py`

- `_cached_resnet50_weights`：扫描 torch hub 缓存，解决不同 torchvision 版本使用不同权重文件名的问题。
- `ResNet50Layer2.__init__`：构造 ResNet-50，只保留 stem、layer1 和 layer2；因此只保留空间特征，不执行全局池化和 ImageNet 分类头。
- `freeze=True`：关闭 backbone 参数梯度，并在训练时强制保持 BatchNorm 的 eval 状态。
- `forward`：输入 `[B,3,H,W]`，输出 `[B,512,H/8,W/8]`；对于 256 像素输入即为约 `[B,512,32,32]`。

## 4. `fas_moe/model.py`

- `SuperpixelMoEConfig`：定义固定的 128/64/16 层级、512 通道、4 个专家和冻结策略。
- `EqualWeightMoE`：把同一尺度的 token 分别送入四个 MLP；`torch.stack(...).mean(dim=0)` 就是老师要求的“专家输出相加后平均”。
- `_labels_to_feature_grid`：使用最近邻插值把 256x256 标签对齐到 backbone 特征图大小，避免产生不存在的区域编号。
- `pool_regions`：用 `index_add_` 累加每个区域的特征并除以像素计数，得到区域平均池化结果。如果某个很小区域在降采样时没有对应网格点，就取该区域质心处的特征作为确定性回退。
- `SuperpixelMoE.forward`：
  1. 没有传入视图时，对 batch 中每张图在 CPU 上运行 SLIC；
  2. 对输入做 ImageNet 均值方差归一化；
  3. 只调用一次 backbone；
  4. 逐层池化并加入位置编码；
  5. 用 `part_distribution @ part_embedding.weight` 加入软部位编码，三项相加后执行 LayerNorm；
  6. 对每层运行四专家平均；
  7. 对区域 token 求均值得到尺度向量，拼接三层后分类。

## 5. `run_moe.py`

- `load_input` 使用 `fas_moe/io.py` 的 JPG/PNG/NPY 输入适配器。
- `segment_views` 生成可视化和模型需要的区域数据。
- `SuperpixelMoE` 完成一次前向；如果传入 `--checkpoint`，只加载已经训练过的参数。
- `summary.json` 保存区域数量、logits、概率、设备和耗时；没有 checkpoint 时明确写入“概率没有训练意义”。

## 6. `train_moe.py`

- `NpyBinaryFASDataset` 使用内存映射读取 CASIA 的 live/spoof NPY，不把整个数据集一次性复制到 RAM。
- 数据值乘以 `255*255`，修复 CASIA 缓存的双重除 255 问题。
- `WeightedRandomSampler` 按类别频率反比分配采样权重，避免一个 smoke batch 恰好只有 spoof。
- `max_steps=1` 是安全默认值；只有显式增加参数才会进行较长训练。
- checkpoint 同时保存模型参数、层级配置和训练历史，便于下一位成员接着实验。

## 7. 推荐阅读顺序

1. 先运行 `run_moe.py`，查看 `labels_*.npy` 和 `tokens_*.npy`；
2. 阅读 `segmentation.py`，确认区域数量和连通性；
3. 阅读 `backbone.py`，确认共享特征图只计算一次；
4. 阅读 `model.py` 的 `pool_regions` 和 `EqualWeightMoE`；
5. 最后运行 `train_moe.py --max-steps 1`，确认梯度和 checkpoint 流程。
