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

当前版本已验证单图推理、单批次前向/反向和短训练流程，并接入 MediaPipe 人脸部位软编码与可选 SLIC 磁盘缓存。它仍不是最终实验版本，不包含复杂路由、Top-k、OCC-FAS 正式协议或最终 HTER/AUC 结果。

## 1. 项目结构

```text
fas_moe/
  __init__.py       公共 Python API
  checkpoint.py     checkpoint 键、形状和结构配置校验
  io.py             JPG/PNG/NPY 输入和数值范围恢复
  features.py       19 维手工区域特征
  segmentation.py   独立的 SLIC 128/64/16 视图
  backbone.py       共享 ResNet-50 空间特征图
  model.py          区域池化、位置编码、MoE 和分类头
tests/
  test_moe_pipeline.py
  test_checkpoint_data.py
run_moe.py          单图前向和中间结果导出
train_moe.py        NPY 数据集短训练入口
cache_landmarks.py  预生成 Landmark 部位缓存
requirements.txt
```

## Landmark 人脸部位编码

当前模型使用 Google MediaPipe Face Landmarker 的 478 点人脸网格生成 11 类互斥部位：

```text
unknown / left_eyebrow / right_eyebrow / left_eye / right_eye /
nose / mouth / left_cheek / right_cheek / forehead / chin
```

官方模型默认放在（以下任一路径均可）：

```text
models/face_landmarker.task
# 或项目根目录的 face_landmarker.task
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

Landmark 版模型新增 `part_embedding` 和 token LayerNorm。`run_moe.py` 会在前向前严格
校验 checkpoint 的参数键、形状和已保存的结构配置；只要这些内容与当前模型不一致，
就会直接报错，正式实验需要使用相同配置重新训练。

三层是同一输入图像的独立 SLIC 空间视图，最终分别修正为 128、64、16 个连通区域。三层之间不保证边界包含关系，也不生成父子映射。

## 2. 从零创建环境

### 2.1 基础要求

- Python 3.10+（Windows 已验证 3.10.16；Linux/WSL 已验证 3.12.3）；
- 推荐安装 Miniconda 或 Anaconda；
- NVIDIA GPU 不是必需条件，但训练时建议使用；
- 如果使用 NVIDIA GPU，驱动必须支持准备安装的 PyTorch CUDA 版本。

当前工作区已验证的参考组合是：

```text
Windows / Conda
  Python       3.10.16
  NumPy        1.26.4
  Pillow       11.3.0
  scikit-image 0.25.2
  PyTorch      2.8.0+cpu
  torchvision  0.23.0+cpu
  MediaPipe    0.10.21
  CUDA         False

Linux / WSL2 Ubuntu 24.04.2
  Python       3.12.3
  NumPy        1.26.4
  Pillow       11.3.0
  scikit-image 0.25.2
  PyTorch      2.8.0+cpu
  torchvision  0.23.0+cpu
  MediaPipe    0.10.21
  matplotlib   3.10.5
  CUDA         False
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

NVIDIA GPU 且准备使用某个 CUDA 版本：

```powershell
python -m pip install torch torchvision --index-url '<PYTORCH_INDEX_URL_FOR_YOUR_CUDA>'
```

只使用 CPU：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

其他 CUDA 版本应根据 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择匹配命令。不要把不匹配的 CUDA wheel 强行安装到环境中。

### 2.4 安装其余依赖

进入 ZIP 解压后的项目目录：

```powershell
Set-Location '<PROJECT_ROOT>'
python -m pip install -r requirements.txt
```

`requirements.txt` 包含当前运行代码需要的 NumPy、Pillow、scikit-image、PyTorch、torchvision、MediaPipe 及其运行时依赖。Windows 上固定 MediaPipe 0.10.21，是因为 0.10.30 的 Tasks ctypes bridge 初始化会报 `function 'free' not found`。Linux/WSL2 使用相同的 Python 依赖版本和官方 CPU torch/torchvision wheel；当前工作区已完成 Linux 端到端验证。当前版本不依赖 `timm`。

检查环境：

```powershell
python -c "import numpy, PIL, skimage, torch, torchvision, mediapipe; print('dependencies: PASS'); print('numpy', numpy.__version__); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('mediapipe', mediapipe.__version__); print('cuda', torch.cuda.is_available())"
```

输出 `dependencies: PASS` 即表示导入成功。CPU 环境显示 `cuda False` 属于正常情况。若使用官方 CPU wheel，请先安装匹配的 torch/torchvision，再安装 `requirements.txt`。

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
| `--landmark-model '<TASK_PATH>'` | MediaPipe `face_landmarker.task` 路径 |
| `--landmark-cache-dir 'outputs/landmark_cache'` | Landmark 缓存目录 |
| `--slic-cache-dir 'outputs/slic_cache'` | SLIC/区域描述缓存目录；默认启用（Python API 传 `None` 可关闭） |
| `--image-range <RANGE>` | NPY RGB 数值范围：`auto`、`0-1/255`、`0-1` 或 `0-255`（推理、训练和缓存脚本均支持） |

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

`train_moe.py` 当前首先支持 NPY 版本的 `domain-generalization` 数据。`--image-range` 支持 `auto`、`0-1/255`、`0-1`、`0-255`；数据集包含极暗图像时建议显式指定。最小训练 smoke 需要以下文件：

```text
<DATASET_ROOT>/
  domain-generalization/
    CASIA-FASD/
      casia_images_live.npy
      casia_images_spoof.npy
```

命令中的 `--dataset-root` 必须指向上面的 `<DATASET_ROOT>`，而不是直接指向 `CASIA-FASD` 文件夹。

数据标签固定为 `live=1`、`spoof=0`。加载器支持 `[0,1/255]`、`[0,1]` 和 `[0,255]`
三种 RGB 数值范围；默认 `--image-range auto` 按数据集整体极值推断。若数据集像素很暗、
导致 `[0,1/255]` 与 `[0,1]` 无法仅凭极值区分，请显式传入 `--image-range 0-1/255` 或
`--image-range 0-1`。数据集不放入代码 ZIP，由每位成员在本地单独准备。

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

测试覆盖 SLIC 数量与连通性、确定性及缓存命中/损坏恢复、三种 NPY 范围恢复、宽整数
范围、19 维区域特征、区域池化、专家平均公式、模型前向/反向、输入范围参数校验，以及 checkpoint 的严格
键/形状/结构配置校验。

### 6.2 运行一个单图样本

```powershell
python run_moe.py `
  --input '<IMAGE_OR_NPY_PATH>' `
  --index 0 `
  --output 'outputs\first_demo' `
  --no-pretrained `
  --landmark-model 'face_landmarker.task' `
  --image-range auto `
  --device cpu
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
  --no-pretrained `
  --landmark-model 'face_landmarker.task' `
  --device cpu `
  --output-dir 'outputs\train_smoke'
```

成功标志：

```text
Training smoke: PASS
```

只有这三步全部通过后，才应开始修改模型或运行长实验。

### 6.4 Linux/WSL 验证

在 Linux 或 WSL 的 Python 3.10+ 环境中，先安装与平台匹配的 CPU/CUDA
PyTorch 和 torchvision，再安装项目依赖。下面命令对应已验证的 Ubuntu 24.04.2 / Python 3.12.3 CPU 环境：

```bash
python3.12 -m pip install torch==2.8.0+cpu torchvision==0.23.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
python3.12 -m pip install -r requirements.txt
python3.12 -m unittest discover -s tests -p 'test_*.py' -v
python3.12 -m compileall -q fas_moe run_moe.py train_moe.py cache_landmarks.py tests
python3.12 run_moe.py \
  --input '<IMAGE_OR_NPY_PATH>' \
  --output 'outputs/linux_smoke' \
  --no-pretrained \
  --landmark-model '<PROJECT_ROOT>/face_landmarker.task' \
  --device cpu
```

验收应看到 `Forward: PASS` 和三层区域数 `128/64/16`；当前 Linux 验证还通过了 18 项
单元测试和 `compileall`。MediaPipe 可能打印 EGL/llvmpipe 图形加速警告，这不影响 CPU
结果。若使用其他 Python 版本或 CUDA wheel，请安装对应匹配的 PyTorch/torchvision 组合。

## 7. 推理与训练输出

`run_moe.py` 输出：

```text
input.png
labels_128.npy / labels_064.npy / labels_016.npy
features_128.npy / features_064.npy / features_016.npy
positions_128.npy / positions_064.npy / positions_016.npy
edges_128.npy / edges_064.npy / edges_016.npy
part_distribution_128.npy / part_distribution_064.npy / part_distribution_016.npy
tokens_128.npy / tokens_064.npy / tokens_016.npy
summary.json
```

- `labels`：`256x256` 连续区域编号；
- `features`：19 维 RGB、Lab、梯度、面积、中心和形状统计；
- `positions`：区域中心、面积比例和包围盒尺度；
- `edges`：无向区域邻接边 `[E,2]`；
- `part_distribution`：每个区域的 11 类人脸部位软分布 `[K,11]`；
- `tokens`：加入位置编码并经过四专家平均后的区域 token；
- `summary.json`：输入信息、SLIC 参数、区域数、logits、概率、设备和耗时；另记录 `slic_cache_hit`、`landmark_cache_hit`、模型路径和 `checkpoint_validation` 报告。

没有训练 checkpoint 时，`summary.json` 中的概率只用于检查数据流。传入 `--checkpoint` 时会先严格校验所有参数键、形状及结构配置（区域层级、通道、位置维度、专家隐藏维度、专家数、类别数、Landmark 开关和输入范围）；不匹配会在前向前报错，不会继续生成 logits。

`train_moe.py` 输出：

```text
checkpoint.pt
history.json
config.json
```

默认 `--max-steps 1` 是防止误启动长训练的安全设置。长实验必须显式增加 `--max-steps` 或 `--epochs`，并使用新的输出目录。

## 8. Python API 和模型接口

```python
import torch
from fas_moe import SuperpixelMoE, SuperpixelMoEConfig, segment_views

# image is one HWC uint8 RGB NumPy array with shape [H,W,3].
views = segment_views(image)
model = SuperpixelMoE(SuperpixelMoEConfig())
images = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
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

批量调用时，`views` 应传入与 batch 等长的 `list[SuperpixelViews]`；单个 `SuperpixelViews`
也可复用于整个 batch。输入图像应为 RGB `[0,255]` 或 `[0,1]` 张量，模型会在内部统一
转换到 `[0,255]` 后做 ImageNet 标准化。

## 9. 继续开发时的约束

修改代码前先阅读本 README。修改后必须满足：

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
cache_landmarks.py
requirements.txt
README.md
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

## 11. 已完成的本轮验收

本轮已完成并验证：Windows/Conda Python 3.10.16 CPU 环境，以及 WSL2 Ubuntu 24.04.2 /
Python 3.12.3 CPU 环境；两端均通过 18 项单元测试、compileall 和三尺度前向 smoke，
并完成真实人脸的 MediaPipe 检测、默认根目录模型发现、SLIC/Landmark 缓存命中与损坏
恢复、严格 checkpoint 校验。随机 backbone 或未训练分类头的概率不代表 FAS 指标。仍可继续扩展：

1. 接入 `domain-generalization-multi` 的 profile/depth/IR；
2. 接入 OCC-FAS 官方 train/dev/test 协议；
3. 增加 MediaPipe 关键点位置编码并做消融；
4. 比较等权专家、软路由和 Top-k 路由；
5. 增加 HTER/AUC、跨域实验和可复现实验配置。

