# FAS Superpixel-MoE 最简可运行基线

本项目实现一条可以继续扩展的 FAS（Face Anti-Spoofing）基线：

```text
RGB 图像
  -> 一次 ResNet-50 空间特征提取
  -> 同一图像的 SLIC 128/64/16 三种视图
  -> 区域平均池化
  -> 几何位置编码
  -> 每个尺度四个专家等权相加后取平均
  -> 三尺度特征拼接
  -> live/spoof 二分类
```

当前版本用于验证单图推理、单批次前向/反向和短训练流程。它还不是最终实验版本，不包含复杂路由、Top-k、关键点编码、全量缓存、OCC-FAS 正式协议或最终 HTER/AUC 结果。

## 1. 项目结构

```text
fas_moe/
  __init__.py       公共 Python API
  io.py             JPG/PNG/NPY 输入和数值范围恢复
  features.py       19 维手工区域特征
  segmentation.py   独立的 SLIC 128/64/16 视图
  backbone.py       共享 ResNet-50 空间特征图
  model.py          区域池化、位置编码、MoE 和分类头
tests/
  test_moe_pipeline.py
run_moe.py          单图前向和中间结果导出
train_moe.py        NPY 数据集短训练入口
requirements.txt
CODE_WALKTHROUGH_MOE.md
```

## Landmark 人脸部位编码

当前模型使用 Google MediaPipe Face Landmarker 的 478 点人脸网格生成 11 类互斥部位：

```text
unknown / left_eyebrow / right_eyebrow / left_eye / right_eye /
nose / mouth / left_cheek / right_cheek / forehead / chin
```

官方模型默认放在：

```text
models/face_landmarker.task
```

### 下载 Face Landmarker 模型

`face_landmarker.task` 是 Google MediaPipe 提供的预训练模型文件，建议不要直接提交到代码仓库，
而是在部署或首次运行前从 Google 官方地址下载：

- 官方模型页面：<https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/index#models>
- 官方模型文件：<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task>

Windows CMD：

```bat
mkdir models 2>nul
curl.exe -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" -o "models\face_landmarker.task"
```

PowerShell：

```powershell
New-Item -ItemType Directory -Force -Path 'models' | Out-Null
Invoke-WebRequest `
  -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' `
  -OutFile 'models\face_landmarker.task'
```

下载后检查文件：

```powershell
Get-Item 'models\face_landmarker.task' | Select-Object FullName, Length
```

当前官方 float16 模型约为 3.6 MB，但上游文件以后可能变化。若确实准备将模型上传到 GitHub，
请先确认实际文件小于 GitHub 的 100 MB 单文件限制，并核对 Google MediaPipe 模型的许可和
重新分发条款；本项目默认通过 `.gitignore` 排除 `models/*.task`。

也可以在命令中使用 `--landmark-model` 指定其他位置。每个尺度会生成
`part_distribution_128.npy`、`part_distribution_064.npy` 和
`part_distribution_016.npy`，其形状分别为 `[K,11]`，每行和为 1。

检测不到人脸、模型缺失或 MediaPipe 初始化失败时，所有 region 自动回退为
`unknown=1`，前向和训练不会中断。运行状态记录在 `summary.json` 的
`landmarks_detected`、`landmark_reason` 和 `landmark_cache_hit` 字段中。

默认缓存目录是 `outputs/landmark_cache`。训练前也可显式预生成缓存：

```powershell
python cache_landmarks.py `
  --dataset-root 'F:\00Dataset\FAS' `
  --dataset 'CASIA-FASD' `
  --landmark-model 'models\face_landmarker.task'
```

临时禁用 Landmark（用于消融实验）：

```powershell
python train_moe.py --dataset-root 'F:\00Dataset\FAS' --no-landmarks --no-pretrained
```

Landmark 版模型新增 `part_embedding` 和 token LayerNorm，因此旧 checkpoint 只能用于
初始化共有参数；正式实验需要重新训练 Landmark 版本。

三层是同一输入图像的独立 SLIC 空间视图，最终分别修正为 128、64、16 个连通区域。三层之间不保证边界包含关系，也不生成父子映射。

## 2. 从零创建环境

### 2.1 基础要求

- Python 3.10；
- 推荐安装 Miniconda 或 Anaconda；
- NVIDIA GPU 不是必需条件，但训练时建议使用；
- 如果使用 NVIDIA GPU，驱动必须支持准备安装的 PyTorch CUDA 版本。

已经验证过的参考组合是：

```text
Python       3.10.18
NumPy        2.2.6
Pillow       11.0.0
scikit-image 0.25.2
PyTorch      2.5.1
torchvision  0.20.1
CUDA build   12.1
```

### 2.2 创建独立 Conda 环境

环境名称可以自行修改，代码不依赖固定名称：

```powershell
conda create -n fas-superpixel-moe python=3.10 -y
conda activate fas-superpixel-moe
python -m pip install --upgrade pip
```

没有 Conda 时也可以使用 Python `venv`，但项目成员应避免直接向系统 Python 安装依赖。

### 2.3 安装 PyTorch

NVIDIA GPU 且准备使用 CUDA 12.1：

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

只使用 CPU：

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

其他 CUDA 版本应根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择匹配命令。不要把不匹配的 CUDA wheel 强行安装到环境中。

### 2.4 安装其余依赖

进入 ZIP 解压后的项目目录：

```powershell
Set-Location '<PROJECT_ROOT>'
python -m pip install -r requirements.txt
```

`requirements.txt` 包含当前运行代码需要的 NumPy、Pillow、scikit-image、PyTorch 和 torchvision。当前版本不依赖 `timm` 或 `mediapipe`。

检查环境：

```powershell
python -c "import numpy, PIL, skimage, torch, torchvision; print('dependencies: PASS'); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
```

输出 `dependencies: PASS` 即表示导入成功。CPU 环境显示 `cuda False` 属于正常情况。

## 3. 解压位置和路径参数

项目可以解压到任意目录，不要在 Python 源码中写个人绝对路径。每位成员只在命令行中替换以下参数：

| 参数或命令 | 含义 |
|---|---|
| `Set-Location '<PROJECT_ROOT>'` | ZIP 解压后的项目目录 |
| `--input '<IMAGE_OR_NPY_PATH>'` | 要处理的图片或 NPY 文件 |
| `--dataset-root '<DATASET_ROOT>'` | 数据集总目录，见下一节的目录结构 |
| `--output 'outputs\demo'` | 输出目录，建议保留为项目内相对路径 |
| `--checkpoint '<CHECKPOINT_PATH>'` | 已训练的 `checkpoint.pt` |
| `--weights-path '<RESNET50_PATH>'` | 离线 ResNet-50 权重文件 |

示例目录只是占位符，运行前必须替换尖括号中的内容：

```powershell
Set-Location 'D:\projects\superpixel+moe'
python run_moe.py --input 'D:\datasets\FAS\3Dmask\3DMAD_images_live.npy' --index 0 --output 'outputs\demo'
python train_moe.py --dataset-root 'D:\datasets\FAS' --dataset CASIA-FASD --max-steps 1
```

## 4. 数据准备

### 4.1 单图推理

`run_moe.py` 支持：

- JPG、JPEG、PNG、BMP、TIF、TIFF；
- 单张 HWC/CHW NPY；
- 批量 NHWC/NCHW NPY，通过 `--index` 选择样本。

所有输入会统一转换为 `256x256` RGB `uint8`。因此单图流程不要求固定数据集目录结构。

### 4.2 当前训练数据

`train_moe.py` 当前首先支持 NPY 版本的 `domain-generalization` 数据。最小训练 smoke 需要以下文件：

```text
<DATASET_ROOT>/
  domain-generalization/
    CASIA-FASD/
      casia_images_live.npy
      casia_images_spoof.npy
```

命令中的 `--dataset-root` 必须指向上面的 `<DATASET_ROOT>`，而不是直接指向 `CASIA-FASD` 文件夹。

数据标签固定为 `live=1`、`spoof=0`。CASIA RGB 缓存位于 `[0,1/255]`，加载器会自动恢复到 `[0,255]`。数据集不放入代码 ZIP，由每位成员在本地单独准备。

## 5. ResNet-50 权重

默认使用 ImageNet 预训练 ResNet-50，并截取到 `layer2`。输入 `256x256` 时共享特征图约为 `[B,512,32,32]`，backbone 默认冻结。

权重加载顺序：

1. `--weights-path` 指定的本地 `.pth`；
2. torch hub 缓存中的 `resnet50-*.pth`；
3. torchvision 自动下载的默认权重。

首次在线运行可能下载约 98 MB 的权重。无法联网时，应由其他成员单独提供兼容的 ResNet-50 权重，例如：

```powershell
python run_moe.py `
  --input '<IMAGE_OR_NPY_PATH>' `
  --weights-path '<RESNET50_PATH>' `
  --output 'outputs\offline_demo'
```

只验证代码结构时可以使用 `--no-pretrained`，但随机 backbone 和未训练分类头输出的概率没有实验意义。

## 6. 接手后的验证顺序

### 6.1 运行单元测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 SLIC 数量与连通性、确定性、输入范围恢复、19 维区域特征、区域池化、专家平均公式、模型前向和反向。

### 6.2 运行一个单图样本

```powershell
python run_moe.py `
  --input '<IMAGE_OR_NPY_PATH>' `
  --index 0 `
  --output 'outputs\first_demo'
```

成功标志：

```text
Levels: {'128': 128, '64': 64, '16': 16}
Forward: PASS
```

### 6.3 运行一个训练 step

```powershell
python train_moe.py `
  --dataset-root '<DATASET_ROOT>' `
  --dataset CASIA-FASD `
  --batch-size 2 `
  --max-steps 1 `
  --output-dir 'outputs\train_smoke'
```

成功标志：

```text
Training smoke: PASS
```

只有这三步全部通过后，才应开始修改模型或运行长实验。

## 7. 推理与训练输出

`run_moe.py` 输出：

```text
input.png
labels_128.npy / labels_064.npy / labels_016.npy
features_128.npy / features_064.npy / features_016.npy
positions_128.npy / positions_064.npy / positions_016.npy
edges_128.npy / edges_064.npy / edges_016.npy
tokens_128.npy / tokens_064.npy / tokens_016.npy
summary.json
```

- `labels`：`256x256` 连续区域编号；
- `features`：19 维 RGB、Lab、梯度、面积、中心和形状统计；
- `positions`：区域中心、面积比例和包围盒尺度；
- `edges`：无向区域邻接边 `[E,2]`；
- `tokens`：加入位置编码并经过四专家平均后的区域 token；
- `summary.json`：输入信息、SLIC 参数、区域数、logits、概率、设备和耗时。

没有训练 checkpoint 时，`summary.json` 中的概率只用于检查数据流。

`train_moe.py` 输出：

```text
checkpoint.pt
history.json
config.json
```

默认 `--max-steps 1` 是防止误启动长训练的安全设置。长实验必须显式增加 `--max-steps` 或 `--epochs`，并使用新的输出目录。

## 8. Python API 和模型接口

```python
from fas_moe import SuperpixelMoE, SuperpixelMoEConfig, segment_views

views = segment_views(image)
model = SuperpixelMoE(SuperpixelMoEConfig())
logits, details = model(images, views=views)
```

主要形状：

```text
输入 images                 [B, 3, 256, 256]
共享特征图                  [B, 512, 32, 32]
details['tokens_128']       [B, 128, 512]
details['tokens_64']        [B, 64, 512]
details['tokens_16']        [B, 16, 512]
输出 logits                 [B, 2]
```

批量调用时，`views` 应传入与 batch 等长的 `list[SuperpixelViews]`。输入图像应为 RGB `[0,255]` 或 `[0,1]` 张量。

## 9. 继续开发时的约束

修改代码前先阅读 `CODE_WALKTHROUGH_MOE.md`。修改后必须满足：

- 三层默认区域数仍是 128、64、16；
- backbone 对一个 batch 只运行一次；
- 不把 SLIC 标签当作图像特征，标签只用于从共享特征图中池化区域；
- `live=1`、`spoof=0` 的标签方向不改变；
- 不把个人数据路径写进 Python 源码；
- 新增依赖必须同步更新 `requirements.txt`；
- 新增功能必须增加对应单元测试；
- 修改 backbone 后应重新训练，旧 checkpoint 不保证兼容。

建议每次实验单独建立输出目录，并记录模型参数、数据来源、随机种子和执行命令。

## 10. ZIP 交接

代码 ZIP 应包含：

```text
fas_moe/
tests/
run_moe.py
train_moe.py
requirements.txt
README.md
CODE_WALKTHROUGH_MOE.md
.gitignore
```

可选加入 `MOE.pptx` 和论文 PDF 作为设计参考。以下内容不要放入代码 ZIP：

```text
outputs/
tmp/
__pycache__/
*.pyc
完整数据集
数据集压缩包
个人 IDE 配置
个人绝对路径配置
```

离线成员需要的 ResNet-50 权重建议单独传输，不与代码版本混在一起。

每次交接应附带：

1. 本次改动的文件；
2. 输入输出接口是否变化；
3. 实际运行过的测试命令和结果；
4. 是否需要新依赖、数据或权重；
5. 当前未解决的问题；
6. 下一位成员建议先完成的任务。

## 11. 当前边界和后续方向

当前代码只证明最简路径可以运行，不代表已经获得有效的 FAS 指标。建议按以下顺序扩展：

1. 为 SLIC 标签增加磁盘缓存，避免每个 epoch 重复计算；
2. 接入 `domain-generalization-multi` 的 profile/depth/IR；
3. 接入 OCC-FAS 官方 train/dev/test 协议；
4. 增加 MediaPipe 关键点位置编码并做消融；
5. 比较等权专家、软路由和 Top-k 路由；
6. 增加 HTER/AUC、跨域实验和可复现实验配置。
