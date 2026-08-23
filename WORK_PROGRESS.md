# 当前工作进度与暂停交接

更新时间：2026-08-23

## 1. 用户要求范围

本轮目标扩展为：

- 审计并修复原项目的低 AUC / 高 HTER 问题；
- 使用 `dataset/domain-generalization` 下 CASIA-FASD、Idiap Replay-Attack、MSU-MFSD 做域内与跨域 LODO；
- 完成阶段 A、B，并新增阶段 C、D、E：
  - A：ResNet-50 + 分类头；
  - B：A + global Native MoE；
  - C：B + Fine/Medium/Coarse 多尺度 SLIC + Shared Naive MoE；
  - D：C + region face-position encoding + Shared Naive MoE；
  - E：D + Fine/Medium/Coarse scale-specific Naive MoE；
- 预留 OULU-NPU 第四域，不伪造本轮未运行的结果；
- 最终清理工作区、整理 Git 结构并写完整复现文档。

## 2. 已完成的代码工作

### 原项目审计发现

1. 旧模型在 forward 内逐图重算 SLIC / MediaPipe，CPU 训练成本过高，实际难以收敛。
2. 旧 `EqualWeightMoE` 只是四个 MLP 的等权平均，没有 router/top-k。
3. 旧训练入口默认配置存在不一致与缺陷：训练配置/数据范围处理脆弱，BatchNorm 与小数据集跨域统计漂移风险高。
4. 旧训练循环缺少 `optimizer.zero_grad()`，并且 `eval_every` 参数丢失，导致阶段 B 批量实验全部失败。
5. 旧四域 manifest 强依赖 OULU-NPU，而当前工作区没有 OULU 文件。
6. 旧阶段 A 在源域验证很好、跨域明显崩溃，证明问题是 source-to-target feature shift，而不是仅仅 threshold。

### 当前代码结构

- `fas_moe/backbone.py`
  - ResNet-50/34 完整卷积主干；
  - 支持 ImageNet 权重与本地权重；
  - 默认冻结 BatchNorm running statistics / eval 状态。
- `fas_moe/model.py`
  - `FASModelConfig` 支持 A/B/C/D/E；
  - `NativeMoE`：top-k router + load-balance loss；
  - `NaiveMoE`：区域 Shared / Scale-Specific 等权专家；
  - `MixStyle`；
  - gradient reversal / domain classifier；
  - 向量化 superpixel region pooling；
  - C：共享区域 MoE；D：位置编码；E：三套尺度 MoE。
- `fas_moe/superpixels.py`
  - SLIC 多尺度缓存；
  - label、位置五维向量、valid mask；
  - cache schema 与损坏缓存恢复。
- `fas_moe/data.py`
  - subject-disjoint split；
  - 三域 intra / LODO manifest；
  - OULU-NPU `pending_dataset` placeholder；
  - cached superpixel views 进入 DataLoader。
- `train_fas.py`
  - A–E 训练入口；
  - batch zero-grad、梯度裁剪；
  - domain/class balanced batch sampler；
  - LODO 默认 strong augmentation、MixStyle、DANN 辅助 loss；
  - validation threshold 选择与严格 checkpoint。
- `evaluate_fas.py`
  - 支持 A–E checkpoint 与 C–E superpixel cache 评测。
- `prepare_data.py`
  - 生成三域 manifest；
  - OULU placeholder；加入 OULU 文件后可重新生成四域协议。
- `run_all_experiments.py`
  - 可恢复的 A–E runner；
  - 跳过已完成目录；
  - 仅在 OULU manifest 为 ready 时加入 OULU 协议。
- `tests/test_fas.py`
  - 8 项测试通过，覆盖 A–E forward/backward、Native/Naive MoE、SLIC cache、region pooling、checkpoint、metrics、manifest、OULU placeholder。

## 3. 已验证结果

### 旧版修复后的阶段 A（历史结果，代码尚未包含本轮最终 DANN/训练循环重构）

这些结果保留在 `outputs/run_all_summary.json` 的历史记录中；本次暂停前已清理 outputs，因此下面是审计记录：

| 协议 | AUC | HTER |
|---|---:|---:|
| A intra_C | 0.9988 | 4.17% |
| A intra_I | 0.9915 | 7.00% |
| A intra_M | 1.0000 | 0.00% |
| A lodo_C | 0.5839 | 42.44% |
| A lodo_I | 0.8305 | 34.07% |
| A lodo_M | 0.8210 | 42.62% |

### 当前重构后的最终 campaign 已完成部分（暂停时）

本次最终 campaign 使用当前 A–E trainer 启动，但在阶段 A 完成、阶段 B 进行中时按用户要求暂停。已直接完成并写入 `outputs` 的阶段 A 结果为：

| 协议 | AUC | HTER | APCER | BPCER |
|---|---:|---:|---:|---:|
| A intra_C | 0.9983 | 2.08% | 0.00% | 4.17% |
| A intra_I | 0.9950 | 11.00% | 2.00% | 20.00% |
| A intra_M | 0.9933 | 6.67% | 3.33% | 10.00% |
| A lodo_C | 0.9205 | 22.44% | 0.22% | 44.67% |
| A lodo_I | 0.8412 | 31.86% | 3.71% | 60.00% |
| A lodo_M | 0.9320 | 25.95% | 1.90% | 50.00% |

阶段 B 已完成的域内结果：

| 协议 | AUC | HTER |
|---|---:|---:|
| B intra_C | 0.9965 | 2.78% |
| B intra_I | 1.0000 | 5.00% |
| B intra_M | 0.9933 | 6.67% |

阶段 B 跨域、阶段 C/D/E 本轮尚未完成，不能声称已达标。

### 当前额外 smoke evidence

在当前最终代码上已做短跑：

- B LODO-C，5 epochs / reduced experts：AUC 0.9309，HTER 21.67%；
- C LODO-C，1 epoch / reduced experts：AUC 0.8517，HTER 24.00%；
- D LODO-C，1 epoch / reduced experts：AUC 0.8259，HTER 26.56%；
- E LODO-C，1 epoch / reduced experts：AUC 0.8121，HTER 24.67%；
- 这些 smoke 不是最终 30 epoch 结果，只用于确认代码通路和优化方向。
- B pilot 的 target oracle HTER 约 12.44%，说明当前跨域模型仍有优化空间，锁定源域 threshold 是主要校准瓶颈之一，但不能用 target oracle 替代正式结果。

## 4. 当前暂停状态

- 已停止所有长时间实验进程；`experiments_final` 已退出，不应继续占用 CPU。
- 已删除本地产出的 `outputs/phaseA..E`、superpixel cache、smoke checkpoint 与 summary，避免把大二进制和临时文件提交 Git。
- 仍保留代码、测试、README、manifest 生成脚本和 OULU placeholder 逻辑。

## 5. 尚未完成

1. 从当前最新代码重新跑完整 A–E × 6 个三域协议（当前历史 outputs 已清理，不能复用旧 checkpoint）。
2. 取得阶段 B 的三域跨域最终结果。
3. 取得阶段 C、D、E 的域内与跨域最终结果。
4. 对任何 HTER > 20% 的 LODO 结果继续优化；当前已加入 MixStyle、DANN、domain/class balanced batches、低 backbone LR、梯度裁剪，但还没有用完整 A–E campaign 证明达标。
5. 加入真实 OULU-NPU 后重新执行四域 manifest 与四域测试；当前仅完成代码占位，未运行 OULU。
6. 运行最终清理后的全套测试/compile，并检查 Git 状态。
7. 当前尚未创建 Git commit；用户要求的“可提交结构”已整理，但 commit 应在后续最终验证后创建。

## 6. 下一步恢复命令

```powershell
# 重新准备三域与 OULU placeholder
python prepare_data.py --dataset-root dataset --output-dir splits

# 先快速确认 B/C/D/E 跨域方向
python run_all_experiments.py --phases BCDE --epochs 5 --batch-size 32

# 通过 smoke 后跑正式三域 A–E
python run_all_experiments.py --phases ABCDE --epochs 30 --batch-size 32

# 结果汇总
python summarize_results.py
python summarize_results.py --json

# 最终验证
python -m unittest discover -s tests -v
python -m compileall -q fas_moe train_fas.py evaluate_fas.py prepare_data.py run_all_experiments.py summarize_results.py tests
```

## 7. 结果解释约束

- `metrics_test.json` 的 threshold 必须来自 source validation；
- 不得用 target labels 选模型、选 threshold 或调参后再报告为正式结果；
- HTER > 20% 必须如实记录并继续优化，不能用 target oracle HTER 冒充正式成绩；
- OULU 文件不存在时必须标记 pending，不得生成伪造的四域数字。
