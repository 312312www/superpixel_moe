# A--E 消融实验运行指南

本指南使用本地 RGB NPY 数据：

```text
F:\00Dataset\FAS\domain-generalization
```

命令中的数据根目录填写 `F:\00Dataset\FAS`。协议属于本地预处理、按 Subject 隔离的
LODO 协议，不应称为四个原始数据集的官方协议；评估单位是本地 NPY 的一个数组条目。

## 1. 当前模型定义

| 模式 | Superpixel | Landmark | 专家 |
|---|---|---|---|
| A | 无 | 无 | 无 |
| B | 无 | 无 | 一套全局 4 专家 |
| C | 128/64/16 | 无 | 三尺度共享一套 4 专家 |
| D | 128/64/16 | 有 | 三尺度共享一套 4 专家 |
| E | 128/64/16 | 有 | 三尺度三套独立 4 专家 |

所有模式使用 ResNet-50 stem--layer2。卷积参数参与训练；BatchNorm 始终保持 eval，
running mean/variance 和 affine 参数均冻结。

## 2. 生成固定 Manifest 和重复审计

在 CMD 中执行：

```bat
cd /d F:\moe\superpixel_moe_upload
python prepare_ablation.py --dataset-root F:\00Dataset\FAS --output-dir outputs\ablation_splits
```

输出：

```text
outputs\ablation_splits\OCI_M.json
outputs\ablation_splits\OMI_C.json
outputs\ablation_splits\OCM_I.json
outputs\ablation_splits\ICM_O.json
outputs\ablation_splits\split_manifest.json
outputs\ablation_splits\duplicate_audit.json
```

实际检查发现 13 组同身份、同标签、同 split 重复；跨身份、跨标签、跨 split 均为 0。

## 3. 预生成缓存

正式运行 C/D/E 前，对 CASIA、Replay、MSU、OULU 分别运行 `cache_landmarks.py`。
该入口在生成 Landmark/部位分布时也会填充 SLIC 缓存。正式 Manifest 训练默认禁止
cache miss；缓存不完整会直接终止，而不是在某个 epoch 临时重算。

## 4. 单步正式管线检查

A 模式不需要区域缓存：

```bat
python train_moe.py --dataset-root F:\00Dataset\FAS --manifest outputs\ablation_splits\OCI_M.json --experiment A --batch-size 6 --epochs 1 --max-steps 1 --image-range 0-1/255 --skip-test --output-dir outputs\preflight\OCI_M\A\seed_7
```

C/D/E 缓存完整后使用同一命令，只修改 `--experiment` 和输出目录。正式模式默认强制
读取缓存；只有调试时才允许显式添加 `--allow-cache-miss`。

## 5. 正式训练输出

每次 Manifest 训练保存：

```text
best_auc.pt
last.pt
config.json
history.json
metrics_val.json
metrics_test.json（未使用--skip-test时）
batch_order.json
manifest.json
environment.json
git_commit.txt
```

Checkpoint 按源域验证 Macro-AUC 选择；并列时选择 Macro-HTER 更低者。目标测试阈值
来自源域验证集最小 Macro-HTER；阈值并列时固定选择较大阈值。目标域 EER 仅作为描述
指标，其阈值不会用于最终分类。

## 6. 仍需用户确认后再锁定的项目

目前不要创建正式 tag 或启动 60 次训练。还需先用 OCI→M 的 A、E 做只查看源域验证的
收敛预实验，再确定 Epoch 和 Scheduler。Batch size 6/12、容量匹配实验 F 都属于建议项，
不是当前 A--E 代码正确运行的前置条件。
