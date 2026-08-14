# Windows 数据集测试结果

测试日期：2026-08-13（Asia/Shanghai）  
工作区：`C:\Users\10086\Downloads\superpixel_moe-main`  
平台：Windows x86-64，Conda Python 3.10.16  
模型：`face_landmarker.task`（项目根目录）

## 结论

四个数据包均能读取，压缩包 CRC 校验均通过。项目当前入口对抽取的代表样本全部完成单样本前向：22/22 条命令退出码为 0，均打印 `Forward: PASS`，三层区域数均为 `128/64/16`，MediaPipe 均为 `landmarks_detected=True`、`landmark_reason=ok`。

`domain-generalization` 的四个 NPY 域还分别完成了一个 batch、一个优化 step 的训练 smoke：4/4 `Training smoke: PASS`。这不是模型精度或 HTER/AUC 评估；推理使用 `--no-pretrained`，且没有训练好的分类 checkpoint，输出概率仅用于检查数据流。

## 测试环境

| 项目 | 实际值 |
|---|---|
| Python | 3.10.16 (`D:\conda\envs\py3.10\python.exe`) |
| NumPy | 1.26.4 |
| Pillow | 11.3.0 |
| scikit-image | 0.25.2 |
| PyTorch | 2.8.0+cpu |
| MediaPipe | 0.10.21 |
| 设备 | CPU (`torch.cuda.is_available() == False`) |
| 模型文件 | `face_landmarker.task`，3,758,596 bytes |

## 数据包盘点

以下统计直接来自 ZIP 中央目录；没有完整解压 2.8 GB 的压缩包。四个 ZIP 均执行 `ZipFile.testzip()`，结果为 `None`。

| ZIP | 压缩大小 | 文件数 | 解压后大小 | 内容与数量 |
|---|---:|---:|---:|---|
| `3Dmask.zip` | 297,679,684 B | 10 | 1,231,399,093 B | 9 个 NPY + `Readme.txt`；3DMAD RGB live/spoof `166/85`，CASIA-3D `288/858`，HKBUv1+ `103/60`；另有 3 个 live depth NPY |
| `domain-generalization.zip` | 2,787,865,201 B | 81 | 16,883,448,765 B | CASIA-FASD `150/450`，Idiap Replay-Attack `140/700`，MSU-MFSD `70/210`，OULU-NPU `990/3960`（live/spoof），RGB 均 `(N,256,256,3)` `float32` |
| `domain-generalization-multi.zip` | 454,507,656 B | 47,224 | 462,651,901 B | CeFA real/spoof `1,797/5,278`，SURF `1,000/5,987`，WMCA `347/1,332`；每类 profile/depth/ir JPG，含 subject-split 说明 |
| `OCC-FAS.zip` | 46,811,740 B | 5,789 | 46,463,022 B | 1,920 个 RGB JPG、3,840 个 256×256 灰度 PNG map、27 个协议 TXT；LF/LF-RO/SF/SF-SO/SF-RO/SF-LF-RO 为 `80/80/320/480/480/480` |

关键范围检查：`3Dmask` RGB NPY 为 `float32 [0,255]`；`domain-generalization` RGB NPY 为 `float32 [0,1/255]`，项目自动识别为 `0-1/255`；图片均按 `0-255` 读取。完整 RGB 数组检查的极值均有限且无 NaN/Inf。

## 代表样本前向测试

每条命令只从 ZIP 抽取一个样本到临时目录，使用 CPU、随机 backbone，冷 SLIC/Landmark 缓存：

```powershell
D:\conda\envs\py3.10\python.exe run_moe.py `
  --input '<SAMPLE>' --index 0 --output '<TEMP_OUTPUT>' `
  --no-pretrained --landmark-model face_landmarker.task --device cpu
```

| 数据包/类别 | 样本覆盖 | 结果 |
|---|---|---|
| 3Dmask | 3DMAD、CASIA-3D、HKBUv1+ 各 live + spoof（6） | 6/6 PASS；NPY range `0-255` |
| domain-generalization | CASIA、Replay、MSU、OULU 各 live + spoof（8） | 8/8 PASS；NPY auto range `0-1/255` |
| domain-generalization-multi | CeFA、SURF、WMCA 各 profile real + spoof（6） | 6/6 PASS；JPG 自动 range `0-255` |
| OCC-FAS | LF live、SF spoof（2） | 2/2 PASS；JPG 自动 range `0-255` |
| **合计** | **22 个输入** | **22/22 exit 0，Forward PASS** |

所有样本均转换为 `256×256×3`，区域计数精确为 `128/64/16`；MediaPipe 均检测到人脸并返回 `ok`。每个样本的 `summary.json` 在清理前均记录 `slic_cache_hit=false`、`landmark_cache_hit=false`；因此这里是冷缓存兼容性 smoke，不代表全量数据集指标。

## domain-generalization 训练入口 smoke

使用从 ZIP 各抽取一张 live 和一张 spoof 构造的最小目录，验证 `train_moe.py` 的真实路径契约、范围恢复、DataLoader、反向传播和 checkpoint 写出：

```powershell
D:\conda\envs\py3.10\python.exe train_moe.py `
  --dataset-root '<TEMP_ROOT>' --dataset '<DATASET>' `
  --batch-size 2 --limit-samples 2 --max-steps 1 `
  --no-pretrained --landmark-model face_landmarker.task `
  --device cpu --output-dir '<TEMP_TRAIN_OUTPUT>'
```

| 数据集 | batch | step | loss（记录值） | 结果 |
|---|---:|---:|---:|---|
| CASIA-FASD | 2 | 1 | 0.7239080667 | `Training smoke: PASS` |
| Idiap Replay-Attack | 2 | 1 | 0.7087379694 | `Training smoke: PASS` |
| MSU-MFSD | 2 | 1 | 0.6679884195 | `Training smoke: PASS` |
| OULU-NPU | 2 | 1 | 0.7134389877 | `Training smoke: PASS` |

这里验证的是入口和梯度链路，不是用 2 张图片训练出的可用模型。完整数据集的正式训练仍需按官方协议划分，并使用训练好的 checkpoint 做评估。

## 协议与模态检查

- `domain-generalization-multi` 的 CeFA/SURF/WMCA 各 real/spoof 的 profile、depth、ir basename 集合一致；代表 JPG 可由 Pillow `verify()` 和完整 RGB 转换读取。该包没有项目当前 `train_moe.py` 所需的 NPY 目录，因此只做图片前向和协议结构检查。
- `OCC-FAS` 的 27 个协议 TXT 所引用路径全部存在（missing=0）；代表 JPG 和 occlusion/spoof PNG 均可解码，尺寸为 `256×256`。它的官方协议与项目 `live=1/spoof=0` 训练入口不同，当前只做图片前向/协议完整性检查，不把它误当作 `train_moe.py` 的 NPY 数据集。
- `3Dmask` 的 RGB NPY 可以直接交给 `run_moe.py`，但目录和命名不符合当前 `train_moe.py` 的四个数据集契约；depth 数组也不是当前 RGB 输入接口，未强行接入训练。

## 可复现的基础回归

测试过程中还运行了：

```powershell
D:\conda\envs\py3.10\python.exe -m unittest discover -s tests -p "test_*.py" -q
D:\conda\envs\py3.10\python.exe -m compileall -q fas_moe run_moe.py train_moe.py cache_landmarks.py tests
D:\conda\envs\py3.10\python.exe -m pip check
```

结果分别为：18/18 tests passed、`compileall` exit 0、`pip check` 输出 `No broken requirements found.`

## 清理记录与限制

- 测试用的抽样 NPY/JPG、临时输出、SLIC/Landmark 缓存和训练 checkpoint 已在报告生成后删除。
- `dataset/*.zip` 和项目根目录的 `face_landmarker.task` 保留；它们不是临时测试产物。
- 本报告的代表样本测试覆盖每个数据包的主要类别/域，但没有逐帧执行完整数据集推理，也没有报告准确率、HTER、AUC 或正式协议成绩。
- `run_moe.py --no-pretrained` 只用于确认数据流和平台兼容性；没有训练 checkpoint 时的概率不具有实验解释力。
