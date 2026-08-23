# FAS Backbone + Native/Naive Multi-Scale MoE

人脸活体检测（live=1 / spoof=0）研究代码，支持当前三个数据集的域内与跨域 LODO 评测，
并为 OULU-NPU 第四域保留可执行的协议占位。

> **当前状态**：本轮长时间实验已按协作要求暂停。代码、测试、A–E 入口、三域协议生成和
> OULU 占位已经完成；完整 A–E × 三域最终数字尚未全部跑完。精确的完成项、已验证数字、
> 未完成项和恢复命令见 [WORK_PROGRESS.md](WORK_PROGRESS.md)。不要把未完成阶段的结果写成已达标。

## 阶段定义

| 阶段 | 结构 |
|---|---|
| A | ImageNet 预训练 ResNet-50 + 二分类头 |
| B | A + global Native MoE：router、top-k=2、8 experts、load-balance loss |
| C | B + Fine/Medium/Coarse 多尺度 SLIC + Shared Naive MoE |
| D | C + 五维 region face-position encoding + Shared Naive MoE |
| E | D + Fine/Medium/Coarse 三套 scale-specific Naive MoE |

C–E 的 SLIC 视图按样本缓存，不在每个 epoch 重新计算。BatchNorm 默认固定为 eval，避免小数据集
和跨域训练时 running statistics 漂移。LODO 默认使用 strong augmentation、source domain/class
balanced batches、MixStyle 和 DANN 辅助域损失；target 标签不参与训练、模型选择或 threshold 选择。

## 1. 环境

```powershell
conda create -n fas python=3.10 -y
conda activate fas
# CPU:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU 请按 PyTorch 官网选择匹配 CUDA wheel
python -m pip install -r requirements.txt
```

依赖：NumPy、Pillow、PyTorch、torchvision、scikit-image。

## 2. 数据目录

当前已验证的三个域：

```text
<DATASET_ROOT>/domain-generalization/
  CASIA-FASD/
    casia_images_live.npy
    casia_images_spoof.npy
    casia_subject_live.npy
    casia_subject_spoof.npy
  Idiap Replay-Attack/
    replay_images_live.npy
    replay_images_spoof.npy
    replay_subject_live.npy
    replay_subject_spoof.npy
  MSU-MFSD/
    MSU_images_live.npy
    MSU_images_spoof.npy
    MSU_subject_live.npy
    MSU_subject_spoof.npy
```

RGB 数组应为 `(N,H,W,3)` 数值数组。当前工作区数据是 float32 `[0,1/255]`，代码会恢复为
`[0,255]`。如果数据来自其他导出器，先检查数值范围，再修改数据读取策略，不要静默错误缩放。

### OULU-NPU 预留

第四域约定文件名：

```text
OULU-NPU/
  Oulu_images_live.npy
  Oulu_images_spoof.npy
  Oulu_subject_live.npy
  Oulu_subject_spoof.npy
```

当前缺少 OULU 文件时，`prepare_data.py` 生成 `intra_O.json`、`lodo_O.json` 和
`four_domain_protocol.json`，状态是 `pending_dataset`，不会伪造四域结果。放入 OULU 文件后
重新生成 manifest，原占位会变成 ready；随后可运行 `lodo_O` 及包含 OULU 的四域协议。

## 3. 生成协议

```powershell
python prepare_data.py --dataset-root '<DATASET_ROOT>' --output-dir splits
```

三域 manifest：

- `intra_C/I/M.json`：各域 subject-disjoint train/val/test，约 70/15/15；
- `lodo_C.json`：Replay + MSU → CASIA；
- `lodo_I.json`：CASIA + MSU → Replay；
- `lodo_M.json`：CASIA + Replay → MSU。

加入 OULU 后，重新执行同一命令；脚本会生成 `intra_O.json`、`lodo_O.json`，并更新已有 LODO
manifest，使其他域的 source 集合包含 OULU。生成的 JSON 含本机绝对路径，因此 `splits/` 被
`.gitignore` 排除，协作者必须在本机重新生成。

## 4. 单次训练

```powershell
# 阶段 A / 域内
python train_fas.py `
  --manifest splits/intra_C.json --phase A `
  --output-dir outputs/phaseA/intra_C

# 阶段 B / 跨域；LODO 推荐 strong augmentation（默认会启用 MixStyle/DANN）
python train_fas.py `
  --manifest splits/lodo_C.json --phase B --augment strong `
  --output-dir outputs/phaseB/lodo_C

# 阶段 C、D、E
python train_fas.py --manifest splits/intra_C.json --phase C --output-dir outputs/phaseC/intra_C
python train_fas.py --manifest splits/intra_C.json --phase D --output-dir outputs/phaseD/intra_C
python train_fas.py --manifest splits/lodo_C.json --phase E --augment strong --output-dir outputs/phaseE/lodo_C
```

关键参数：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 32 | batch 大小 |
| `--lr` | 5e-5 | 新增模块学习率 |
| `--backbone-lr` | `lr*0.2` | ResNet 微调学习率 |
| `--weight-decay` | 1e-3 | AdamW 正则 |
| `--label-smoothing` | 0.2 | 标签平滑 |
| `--num-experts` / `--top-k` | 8 / 2 | Native MoE 与 Naive MoE 专家配置 |
| `--augment` | standard | LODO 推荐 `strong` |
| `--mixstyle-prob` | LODO 自动 0.5 | feature style 混合概率 |
| `--domain-loss-weight` | LODO 自动 0.05 | DANN 域对抗损失权重 |
| `--domain-balanced-batches` | true | source domain/class 均衡批次 |
| `--superpixel-cache-dir` | `outputs/superpixel_cache` | C–E SLIC cache |

每个实验目录会保存：

```text
best_auc.pt        验证 AUC 最优 checkpoint（threshold 来自 source validation）
metrics_val.json   验证指标
metrics_test.json  测试指标
history.json       每 epoch 曲线
config.json        完整配置、协议和环境信息
```

## 5. 可恢复的一键 runner

runner 会跳过已经存在 `metrics_test.json` 的实验，适合 CPU 长任务中断后恢复；只在 OULU
manifest 状态为 ready 时加入 OULU 协议。

```powershell
# 完整 A–E × 当前三域协议
python run_all_experiments.py --phases ABCDE --epochs 30 --batch-size 32

# 只跑新增阶段，适合先接续 A/B 后验证 C/D/E
python run_all_experiments.py --phases CDE --epochs 30 --batch-size 32

# 强制覆盖已完成结果
python run_all_experiments.py --phases ABCDE --force
```

加入 OULU 后：

```powershell
python prepare_data.py --dataset-root '<DATASET_ROOT>' --output-dir splits
python run_all_experiments.py --phases ABCDE --epochs 30 --batch-size 32
```

## 6. 评测与汇总

```powershell
python evaluate_fas.py `
  --checkpoint outputs/phaseE/lodo_C/best_auc.pt `
  --manifest splits/lodo_C.json --split test `
  --output-dir outputs/eval_phaseE_lodo_C

python summarize_results.py
python summarize_results.py --json
```

指标：

- AUC：live=1 的 ROC AUC；
- APCER：spoof 被判 live 的比例；
- BPCER：live 被判 spoof 的比例；
- HTER/ACER：`(APCER+BPCER)/2`；
- EER：描述性指标；正式 test threshold 仍锁定为 source validation 最小 macro-HTER。

## 7. 已验证结果边界

暂停前，当前重构 trainer 已完整跑完阶段 A 六个三域协议，以及阶段 B 三个域内协议：

| 阶段 | 协议 | AUC | HTER |
|---|---|---:|---:|
| A | intra_C | 0.9983 | 2.08% |
| A | intra_I | 0.9950 | 11.00% |
| A | intra_M | 0.9933 | 6.67% |
| A | lodo_C | 0.9205 | 22.44% |
| A | lodo_I | 0.8412 | 31.86% |
| A | lodo_M | 0.9320 | 25.95% |
| B | intra_C | 0.9965 | 2.78% |
| B | intra_I | 1.0000 | 5.00% |
| B | intra_M | 0.9933 | 6.67% |

阶段 B 跨域、C/D/E 全协议结果在暂停时尚未完成。上述 LODO A 仍有 HTER>20%，因此下一轮
必须继续优化并如实记录；不能用 target oracle threshold 冒充正式结果。完整细节和 smoke 证据
见 [WORK_PROGRESS.md](WORK_PROGRESS.md)。

## 8. 测试与 Git 协作

```powershell
python -m unittest discover -s tests -v
python -m compileall -q fas_moe train_fas.py evaluate_fas.py prepare_data.py run_all_experiments.py summarize_results.py tests
```

当前测试覆盖 A–E forward/backward、Native/Naive MoE、SLIC cache、region pooling、位置编码、
checkpoint、metrics、subject split、三域 manifest 和 OULU placeholder。

可提交的源代码结构：

```text
fas_moe/                 模型、数据、指标、checkpoint、SLIC cache
train_fas.py             A–E 训练入口
prepare_data.py          manifest 生成与 OULU placeholder
evaluate_fas.py          checkpoint 评测
run_all_experiments.py   可恢复 runner
summarize_results.py     结果表
 tests/test_fas.py       单元测试
README.md                复现流程
WORK_PROGRESS.md         本轮暂停交接
requirements.txt         依赖
.gitignore               数据/输出/缓存排除规则
```

提交前清理本地生成物：

```powershell
Remove-Item -Recurse -Force outputs, splits, __pycache__, fas_moe\__pycache__, tests\__pycache__ -ErrorAction SilentlyContinue
python -m unittest discover -s tests -v
```

`dataset/`、`outputs/`、`splits/`、SLIC cache、checkpoint、模型权重和 Python 缓存均不进入 Git；
manifest 可由命令重新生成。当前暂停阶段不自动创建 commit，待下一轮最终 A–E/OULU 验证通过后再提交。
