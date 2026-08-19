# Superpixel-MoE 下载、安装与运行指南（Windows）

本文说明其他成员如何从 GitHub 下载项目，配置 CPU 或 NVIDIA GPU 环境，安装 MediaPipe Face Landmarker，运行测试、单图推理、Landmark 缓存和训练。

> 当前 `train_moe.py` 仍是基础训练入口，默认只运行 1 个 step。它可用于验证训练链路，但尚未包含正式的训练集/验证集/测试集协议、最佳模型选择和 ACER/HTER/AUC 评估。未加载经过 Landmark 版本重新训练的 checkpoint 时，推理概率没有实际分类意义。

## 1. 准备软件

推荐环境：

- Windows 10/11；
- Git；
- Miniconda 或 Anaconda；
- Python 3.10；
- 可选：支持 CUDA 的 NVIDIA GPU。

检查 Git 和 Conda：

```powershell
git --version
conda --version
```

如果命令不存在，请先安装：

- Git：<https://git-scm.com/download/win>
- Miniconda：<https://docs.conda.io/projects/miniconda/en/latest/>

安装完成后重新打开 PowerShell。

## 2. 下载项目

选择一个保存代码的位置，例如 `D:\projects`：

```powershell
New-Item -ItemType Directory -Force -Path 'D:\projects' | Out-Null
Set-Location 'D:\projects'
git clone https://github.com/312312www/superpixel_moe.git
Set-Location 'D:\projects\superpixel_moe'
```

确认当前分支和文件：

```powershell
git branch --show-current
Get-ChildItem
```

正常情况下分支为 `main`，并能看到：

```text
fas_moe/
tests/
run_moe.py
train_moe.py
cache_landmarks.py
requirements.txt
```

以后更新代码时，在项目目录执行：

```powershell
git pull origin main
```

## 3. 创建独立 Python 环境

```powershell
conda create -n fas-superpixel-moe python=3.10 -y
conda activate fas-superpixel-moe
python -m pip install --upgrade pip
```

确认当前解释器：

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Python 应为 3.10，路径应指向 `fas-superpixel-moe` 环境。

## 4. 安装 PyTorch

CPU 和 GPU 只需选择一种安装方式。

### 4.1 CPU 版本

没有 NVIDIA GPU，或只想检查流程时执行：

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

### 4.2 NVIDIA GPU / CUDA 12.1 版本

先检查显卡驱动：

```powershell
nvidia-smi
```

如果驱动支持所需 CUDA 运行时，可安装：

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

不要直接安装仓库外流传的巨大 `.whl` 文件。其他 CUDA 版本请根据 PyTorch 官方安装页面选择匹配命令：

<https://pytorch.org/get-started/locally/>

## 5. 安装项目依赖

确认 PowerShell 当前位于项目根目录：

```powershell
Set-Location 'D:\projects\superpixel_moe'
python -m pip install -r requirements.txt
```

检查关键依赖：

```powershell
python -c "import numpy, PIL, skimage, torch, torchvision, mediapipe; print('dependencies: PASS'); print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())"
```

CPU 环境显示 `cuda: False` 是正常的。GPU 环境预期显示 `cuda: True`。

如果 GPU 没有被识别，执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

确认安装的不是 CPU 版 PyTorch，并检查 NVIDIA 驱动。

## 6. 下载 MediaPipe Face Landmarker 模型

项目默认从以下位置读取模型：

```text
models/face_landmarker.task
```

创建目录并从 Google 官方地址下载：

```powershell
New-Item -ItemType Directory -Force -Path 'models' | Out-Null
Invoke-WebRequest `
  -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' `
  -OutFile 'models\face_landmarker.task'
```

检查模型文件：

```powershell
Get-Item 'models\face_landmarker.task' | Select-Object FullName, Length
```

如果下载失败，也可以在浏览器打开下面的地址，下载后手动放入 `models` 文件夹：

<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task>

该模型文件被 `.gitignore` 排除，不会随 Git 仓库下载，所有使用者都需要单独准备。

## 7. 运行单元测试

```powershell
python -m unittest discover -s tests -v
```

当前预期结果：

```text
Ran 9 tests
OK
```

测试覆盖：

- 128/64/16 三尺度 SLIC 区域数量和连通性；
- Region Pooling；
- 11类人脸部位软分布；
- Landmark 模型缺失时回退到 `unknown`；
- 模型前向和反向传播；
- `part_embedding` 能获得梯度。

只有测试全部通过，才建议继续推理或训练。

## 8. 单张图片推理

准备一张正面或接近正面的人脸图片，例如：

```text
D:\test_images\face.jpg
```

第一次只检查数据流时，可关闭 ImageNet 预训练权重下载：

```powershell
python run_moe.py `
  --input 'D:\test_images\face.jpg' `
  --output 'outputs\first_demo' `
  --landmark-model 'models\face_landmarker.task' `
  --no-pretrained `
  --device cpu
```

预期终端出现：

```text
Levels: {'128': 128, '64': 64, '16': 16}
Forward: PASS
```

注意：`--no-pretrained` 会使用随机初始化的 Backbone；未提供训练 checkpoint 时，输出概率只能证明程序可运行，不能判断真实或攻击。

### 8.1 使用 ImageNet 预训练 Backbone

有网络时，可以去掉 `--no-pretrained`：

```powershell
python run_moe.py `
  --input 'D:\test_images\face.jpg' `
  --output 'outputs\pretrained_demo' `
  --landmark-model 'models\face_landmarker.task' `
  --device auto
```

第一次运行可能自动下载 ResNet-50 权重。

### 8.2 使用已训练 checkpoint

```powershell
python run_moe.py `
  --input 'D:\test_images\face.jpg' `
  --checkpoint 'D:\checkpoints\landmark_checkpoint.pt' `
  --landmark-model 'models\face_landmarker.task' `
  --output 'outputs\checkpoint_demo' `
  --device auto
```

必须使用加入 Landmark 后重新训练的 checkpoint。旧版 checkpoint 缺少 `part_embedding` 和 token LayerNorm 参数，不能作为正式结果使用。

### 8.3 NPY 输入

单张 HWC/CHW NPY：

```powershell
python run_moe.py `
  --input 'D:\data\single_face.npy' `
  --output 'outputs\npy_demo'
```

批量 NHWC/NCHW NPY 中选择第 10 个样本：

```powershell
python run_moe.py `
  --input 'D:\data\faces.npy' `
  --index 10 `
  --output 'outputs\npy_index_10'
```

### 8.4 临时关闭 Landmark

用于消融或排查 MediaPipe 问题：

```powershell
python run_moe.py `
  --input 'D:\test_images\face.jpg' `
  --no-landmarks `
  --output 'outputs\without_landmarks'
```

关闭后所有部位编码不会加入 Region Token。

## 9. 检查推理输出

推理结果位于 `--output` 指定目录，例如：

```text
outputs/first_demo/
```

关键文件：

```text
input.png
summary.json
labels_128.npy / labels_064.npy / labels_016.npy
part_distribution_128.npy
part_distribution_064.npy
part_distribution_016.npy
positions_128.npy / positions_064.npy / positions_016.npy
tokens_128.npy / tokens_064.npy / tokens_016.npy
```

查看 Landmark 是否成功：

```powershell
Get-Content 'outputs\first_demo\summary.json'
```

重点字段：

```json
{
  "landmarks_enabled": true,
  "landmarks_detected": true,
  "landmark_reason": "ok"
}
```

如果 `landmarks_detected` 为 `false`，程序仍会运行，但所有 region 会回退到 `unknown`。此时检查 `landmark_reason`，确认是模型路径错误、MediaPipe 缺失，还是图片中未检测到人脸。

检查软部位分布：

```powershell
python -c "import numpy as np; d=np.load(r'outputs\first_demo\part_distribution_128.npy'); print(d.shape); print('max row-sum error:', abs(d.sum(1)-1).max()); print('non-unknown regions:', (d[:,1:].sum(1)>1e-6).sum())"
```

正常应满足：

- 形状为 `(128, 11)`；
- 每行概率和接近1；
- 成功检测人脸时通常存在非 unknown 区域。

## 10. 准备训练数据

当前训练入口读取 NPY 格式数据，目录必须为：

```text
<DATASET_ROOT>/
  domain-generalization/
    CASIA-FASD/
      casia_images_live.npy
      casia_images_spoof.npy
```

例如：

```text
F:/00Dataset/FAS/
  domain-generalization/
    CASIA-FASD/
      casia_images_live.npy
      casia_images_spoof.npy
```

此时命令中的 `--dataset-root` 应填写：

```text
F:\00Dataset\FAS
```

而不是直接填写 `CASIA-FASD` 文件夹。

数组要求：

- 形状为 NHWC；
- 最后一维为3通道 RGB；
- 当前数据加载器针对项目原有缓存格式 `[0, 1/255]` 做了恢复；
- `live=1`，`spoof=0`。

如果自己的 NPY 是普通 `[0,1]` 或 `[0,255]`，不要直接训练，应先修改/统一数据加载逻辑，否则数值会被错误放大。

## 11. 预生成 Landmark 缓存

训练时每张图片都需要 SLIC 和 Landmark。建议先生成缓存，避免后续重复检测。

先用少量样本验证：

```powershell
python cache_landmarks.py `
  --dataset-root 'F:\00Dataset\FAS' `
  --dataset 'CASIA-FASD' `
  --landmark-model 'models\face_landmarker.task' `
  --cache-dir 'outputs\landmark_cache' `
  --limit-samples 20
```

确认没有路径或模型错误后，再处理完整数据集：

```powershell
python cache_landmarks.py `
  --dataset-root 'F:\00Dataset\FAS' `
  --dataset 'CASIA-FASD' `
  --landmark-model 'models\face_landmarker.task' `
  --cache-dir 'outputs\landmark_cache'
```

结束后会显示：

```text
Processed: ...; detected: ...; cache hits: ...
```

应记录 `detected / processed` 比例。如果检测成功率异常低，应先检查图片质量、输入数值范围和人脸裁剪，再开始训练。

## 12. 运行最小训练验证

先只使用少量样本和1个 step：

```powershell
python train_moe.py `
  --dataset-root 'F:\00Dataset\FAS' `
  --dataset 'CASIA-FASD' `
  --batch-size 2 `
  --epochs 1 `
  --max-steps 1 `
  --limit-samples 20 `
  --landmark-model 'models\face_landmarker.task' `
  --landmark-cache-dir 'outputs\landmark_cache' `
  --output-dir 'outputs\train_smoke' `
  --device auto
```

成功标志：

```text
Training smoke: PASS
```

输出包括：

```text
outputs/train_smoke/checkpoint.pt
outputs/train_smoke/history.json
outputs/train_smoke/config.json
```

## 13. 增加训练步数

当前脚本的 `--max-steps` 是每个 epoch 的最大 step 数。确认最小训练通过后，可以增加参数，例如：

```powershell
python train_moe.py `
  --dataset-root 'F:\00Dataset\FAS' `
  --dataset 'CASIA-FASD' `
  --batch-size 8 `
  --epochs 10 `
  --max-steps 500 `
  --learning-rate 0.0001 `
  --landmark-model 'models\face_landmarker.task' `
  --landmark-cache-dir 'outputs\landmark_cache' `
  --output-dir 'outputs\train_landmark_v1' `
  --device auto
```

但是，该命令只会执行当前基础训练循环。正式研究还需要补充独立验证集、最佳 checkpoint、断点续训、阈值选择和 FAS 指标，因此不要把这里生成的 loss 或 checkpoint 直接视为正式实验结果。

默认 Backbone 冻结。需要训练 Backbone 时可以增加：

```text
--train-backbone
```

训练 Backbone 会明显增加显存和训练时间，也必须重新训练完整模型。

## 14. 使用训练产生的 checkpoint 推理

```powershell
python run_moe.py `
  --input 'D:\test_images\face.jpg' `
  --checkpoint 'outputs\train_landmark_v1\checkpoint.pt' `
  --landmark-model 'models\face_landmarker.task' `
  --landmark-cache-dir 'outputs\landmark_cache' `
  --output 'outputs\trained_inference' `
  --device auto
```

如果出现 checkpoint 架构差异提示，说明 checkpoint 可能来自旧模型或不同配置。正式结果必须使用相同模型结构重新训练的 checkpoint。

## 15. 常见问题

### 15.1 `conda` 无法识别

使用 Anaconda Prompt，或重新打开终端。也可以先执行：

```powershell
conda init powershell
```

然后关闭并重新打开 PowerShell。

### 15.2 `ModuleNotFoundError: No module named 'torch'`

通常是环境未激活或 PyTorch 安装到了另一个 Python：

```powershell
conda activate fas-superpixel-moe
python -c "import sys; print(sys.executable)"
python -m pip show torch
```

使用 `python -m pip` 安装，避免 `pip` 和 `python` 指向不同环境。

### 15.3 MediaPipe 无法导入

```powershell
python -m pip install "mediapipe>=0.10.30,<0.11"
python -c "import mediapipe; print(mediapipe.__version__)"
```

本项目推荐 Python 3.10。若使用较新的 Python 导致没有兼容 wheel，请重新创建 Python 3.10 环境。

### 15.4 Landmark 模型不存在

检查：

```powershell
Test-Path 'models\face_landmarker.task'
```

如果返回 `False`，重新执行第6节的下载命令，或通过 `--landmark-model` 指向实际文件。

### 15.5 `landmarks_detected=false`

查看 `summary.json` 中的 `landmark_reason`：

- `model is not configured`：没有配置模型；
- `model does not exist`：模型路径错误；
- `mediapipe is unavailable`：依赖没有安装；
- `no face detected`：图片中未检测到人脸；
- 其他错误：检查模型文件是否完整。

对 `no face detected`，优先尝试清晰、正面、无遮挡、脸部占比较高的图片。

### 15.6 首次运行下载 ResNet-50 失败

只检查代码时使用：

```text
--no-pretrained
```

正式训练可提前准备兼容的 ResNet-50 权重，然后通过：

```text
--weights-path 'D:\weights\resnet50.pth'
```

指定本地文件。

### 15.7 CUDA 显存不足

依次尝试：

1. 减小 `--batch-size`；
2. 保持 Backbone 冻结，不使用 `--train-backbone`；
3. 使用 `--device cpu` 验证流程；
4. 关闭其他占用显存的程序。

### 15.8 输出概率看起来不合理

确认是否加载了经过 Landmark 版本训练的 checkpoint。以下情况的概率都没有实际意义：

- 使用 `--no-pretrained` 且没有 checkpoint；
- 只有 ImageNet Backbone，但分类头未经 FAS 训练；
- 使用旧版非 Landmark checkpoint；
- 只运行了一个 smoke step。

## 16. 推荐验收顺序

新成员接手项目时建议严格按以下顺序：

1. `git clone` 下载代码；
2. 创建 Python 3.10 Conda 环境；
3. 安装匹配硬件的 PyTorch；
4. 安装 `requirements.txt`；
5. 下载 `models/face_landmarker.task`；
6. 运行9项单元测试；
7. 用单张人脸运行 `run_moe.py --no-pretrained`；
8. 检查 `summary.json` 中 Landmark 是否成功；
9. 检查 `[K,11]` 部位分布；
10. 准备符合要求的 NPY 数据；
11. 预生成 Landmark 缓存；
12. 运行1-step训练；
13. 再决定是否开展更长训练。

如果某一步失败，应先解决当前步骤，不要直接开始长时间训练。
