# CLAUDE.md — 调试记忆与行为记录

> 本文件供 Claude AI 在多轮调试会话中保持上下文记忆。每次调试后更新。

---

## 项目概要

| 字段 | 值 |
|------|-----|
| 项目名 | hnscc-nnunet |
| 类型 | nnUNet v2 医学图像分割（3D CT，头颈鳞状细胞癌肿瘤） |
| 框架 | nnUNet v2 + PyTorch |
| Python | 3.11（虚拟环境 `.venv/`） |
| 工作目录 | `/mnt/d/code/project AI-MED/HNSCC/MAR6_nnunet` |
| 数据集 ID | 100，名称 `HNSCC_CT` |

---

## 环境配置

```bash
# 激活虚拟环境
source .venv/bin/activate   # 或直接用 .venv/bin/python

# 环境变量（train.py 会在导入 nnunetv2 之前自动设置，无需手动 export）
export nnUNet_raw=/mnt/d/code/project\ AI-MED/HNSCC/MAR6_nnunet/nnUNet_raw
export nnUNet_preprocessed=/mnt/d/code/project\ AI-MED/HNSCC/MAR6_nnunet/nnUNet_preprocessed
export nnUNet_results=/mnt/d/code/project\ AI-MED/HNSCC/MAR6_nnunet/nnUNet_results
```

**已安装的关键包：**
- torch 2.10.0+cpu（CPU-only，无 CUDA）
- nnunetv2（已安装，版本标记为 `installed`）
- nibabel 5.4.0
- numpy 2.4.2
- matplotlib 3.10.8
- blosc2 4.1.2

---

## 目录结构

```
MAR6_nnunet/
├── .venv/                        # Python 虚拟环境
├── raw_data/
│   ├── images/                   # 原始 CT NIfTI 文件（*.nii.gz）
│   └── masks/                    # 对应的二值分割掩码
├── nnUNet_raw/
│   └── Dataset100_HNSCC_CT/
│       ├── imagesTr/             # 8 例训练图像（_0000 后缀表示 channel 0）
│       ├── imagesTs/             # 2 例测试图像
│       ├── labelsTr/             # 8 例训练标签
│       └── dataset.json          # 数据集元数据
├── nnUNet_preprocessed/
│   └── Dataset100_HNSCC_CT/
│       ├── nnUNetPlans.json      # 自动生成的训练计划
│       ├── dataset_fingerprint.json
│       ├── nnUNetPlans_2d/       # 2D 预处理文件（.b2nd, .pkl）
│       └── nnUNetPlans_3d_fullres/ # 3D 预处理文件
├── nnUNet_results/
│   └── Dataset100_HNSCC_CT/
│       └── nnUNetTrainer__nnUNetPlans__2d/
│           └── fold_0/           # checkpoint_best.pth / checkpoint_final.pth
├── train.py                      # 完整流程脚本（CPU/GPU 兼容，--images/--masks 接收真实数据）
├── nnunet_3d_ct_training.ipynb   # 完整训练流程 Jupyter Notebook（参考用）
├── pyproject.toml                # 项目依赖配置
└── CLAUDE.md                     # 本文件
```

---

## 主要脚本说明

### `train.py`（唯一入口脚本）

完整流程脚本，CPU / GPU 均可运行。接受真实 NIfTI 数据集路径，一键完成：
1. 输入路径验证（检查文件存在性与 images/masks 文件名配对）
2. nnUNet 数据集格式化（imagesTr / labelsTr / dataset.json，已格式化则跳过）
3. plan_and_preprocess（已完成则自动跳过）
4. 训练（nnUNetTrainer Python API，自动保存 checkpoint）

| 模式 | 配置 | Epoch | Iter/epoch |
|------|------|-------|------------|
| CPU  | 2d   | 5     | 20（smoke test） |
| GPU  | 3d_fullres | 1000 | 250（完整训练） |

**必需参数：**
- `--images`：CT NIfTI 文件目录（*.nii.gz）
- `--masks`：对应 mask NIfTI 目录（*.nii.gz，文件名须与 images 完全匹配）

**可选参数：**
- `--dataset-id`（默认 100）、`--dataset-name`（默认 HNSCC_CT）
- `--fold`（默认 0）、`--train-split`（默认 0.8）
- `--channel`（默认 CT）、`--label`（默认 tumor）
- `--epochs`（覆盖 epoch 数）、`--config`（覆盖 nnUNet 配置）
- `--force-preprocess`（强制重新运行预处理）

```bash
# 基本用法（CPU smoke test）
.venv/bin/python train.py --images raw_data/images --masks raw_data/masks

# 指定数据集参数
.venv/bin/python train.py \
    --images /data/hnscc/images \
    --masks  /data/hnscc/masks  \
    --dataset-id 101 --dataset-name MY_CT --label tumor --epochs 500

# GPU 完整训练
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
    --images /data/hnscc/images --masks /data/hnscc/masks
```

### `nnunet_3d_ct_training.ipynb`
完整交互式训练流程参考，含安装、预处理、训练、推理、评估。

---

## 调试 / 运行历史

### 第 1 轮（2026-03-06）

**操作：** 初次探索项目结构，运行 `debug_pipeline.py`（已删除）

**结果：所有步骤通过**

```
STEP 1 — 环境检查       全部 ✓
STEP 2 — 路径配置       全部 ✓
STEP 3 — 数据集准备     全部 ✓（8 训练 / 2 测试）
STEP 9 — 工具函数       全部 ✓
STEP 4 — plan_and_preprocess  ✓（2D + 3d_fullres，3d_lowres 因尺寸差异被丢弃）
STEP 5-8 — 训练/推理   ~ （需 GPU，跳过）
```

---

### 第 2 轮（2026-03-06）

**操作：** 运行 `debug_stage2.py`（已删除），验证预处理文件 + CPU 前向传播

**结果：所有步骤通过**

```
Step 4b — 预处理文件完整性      全部 ✓（.b2nd / .pkl）
Step 5a — 模型初始化（CPU）     2D 7.77M + 3D 16.14M 参数 ✓
Step 5b — CPU 前向传播          2D + 3d_fullres ✓
```

---

### 第 3 轮（2026-03-06）

**操作：** 清理调试文件，编写 `train.py`，完整跑通 CPU smoke test

**已删除：** `debug_pipeline.py`、`debug_stage2.py`、`debug_slice.png`、`create_mock_data.py`

**结果：全流程跑通**

```
Step 1 — Mock 数据检查     ✓（已有 10 例，跳过生成）
Step 2 — 数据集格式化      ✓（已格式化，跳过复制）
Step 3 — plan_and_preprocess ✓（已完成，跳过）
Step 4 — 训练完成          ✓（2d，5 epoch × 20 iter，CPU，约 60 秒）
  Epoch 0: train_loss=0.4285  val_loss=0.0137  Dice=0.012
  Epoch 4: train_loss=-0.6704 val_loss=-0.7022 Dice=0.769
```

**Checkpoint 保存于：**
`nnUNet_results/Dataset100_HNSCC_CT/nnUNetTrainer__nnUNetPlans__2d/fold_0/`
- `checkpoint_best.pth`（60 MB）
- `checkpoint_final.pth`（60 MB）

**无错误，无需修复。**

---

### 第 4 轮（2026-03-06）

**操作：** 重构 `train.py`，移除 mock 数据生成，改为 CLI 参数接收真实 NIfTI 数据集路径

**变更内容：**
- 删除 Step 1（mock CT 数据生成逻辑）
- 新增 `argparse` 参数解析（`--images`、`--masks` 为必填）
- Step 1 改为输入路径验证（目录存在性 + 文件名配对检查）
- 新增可选参数：`--dataset-id`、`--dataset-name`、`--fold`、`--train-split`、`--channel`、`--label`、`--epochs`、`--config`、`--force-preprocess`
- `nibabel` 导入已移除（不再需要生成 mock 数据）

**无 mock 依赖，脚本更简洁，兼容任意 NIfTI CT 数据集。**

---

## 当前状态

| 组件 | 状态 |
|------|------|
| 环境 + 依赖 | ✓ 完整 |
| train.py（真实数据路径接口） | ✓ 完整（--images / --masks CLI 参数） |
| nnUNet 数据格式化 | ✓ 完整（8 训练 / 2 测试，使用 mock 数据验证） |
| plan_and_preprocess | ✓ 完整（2D + 3d_fullres） |
| CPU smoke test 训练 | ✓ 完整（2D，5 epoch，Dice↑ 0.012→0.769） |
| GPU 完整训练（3d_fullres） | 待切换 CUDA 环境 |
| 推理 | 等待 GPU 训练完成 |
| 评估（Dice / HD95） | 等待推理完成 |

---

## 关键 API 记录

### nnUNetTrainer Python API
```python
# 构造函数签名
nnUNetTrainer(plans: dict, configuration: str, fold: int,
              dataset_json: dict, device: torch.device)

# 实例属性覆盖（必须在 initialize() 之前）
trainer.num_epochs                   = N
trainer.num_iterations_per_epoch     = M
trainer.num_val_iterations_per_epoch = K

trainer.initialize()
trainer.run_training()
```

---

## 切换 GPU 训练

```bash
# 1. 安装 CUDA 版 PyTorch
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 运行 train.py（自动检测 GPU，切换为 3d_fullres + 1000 epoch）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train.py \
    --images /data/hnscc/images --masks /data/hnscc/masks

# 或使用 nnUNet CLI
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 100 3d_fullres 0 -tr nnUNetTrainer

# 推理
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i "nnUNet_raw/Dataset100_HNSCC_CT/imagesTs" \
    -o "predictions" \
    -d 100 -c 3d_fullres -tr nnUNetTrainer \
    -chk checkpoint_best.pth -f 0

# 评估（Dice / HD95）
nnUNetv2_evaluate_folder \
    -gt "nnUNet_raw/Dataset100_HNSCC_CT/labelsTr" \
    -pred "predictions" \
    -djfile "nnUNet_raw/Dataset100_HNSCC_CT/dataset.json" \
    -pfile "nnUNet_results/Dataset100_HNSCC_CT/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"
```

---

## 已知注意事项

1. **路径含空格：** 工作目录路径含空格（`project AI-MED`），shell 命令需加引号。
2. **环境变量时机：** 必须在 `import nnunetv2` 之前设置 `os.environ`；`train.py` 已在文件顶部处理。
3. **文件名配对：** `--images` 和 `--masks` 目录中文件名须完全一致（排序后一一对应）。
4. **fft_conv_pytorch 警告：** `UserWarning: Using a non-tuple sequence for multidimensional indexing` — 来自第三方库，不影响训练结果。
5. **3d_lowres 不存在：** 小尺寸数据，nnUNet 自动丢弃该配置属正常行为。
6. **nnUNet ResEnc planner：** nnUNet 推荐新版 ResEnc 预设，当前使用默认旧 planner，调试无影响。

---

## 下次任务检查清单

- [ ] 准备好真实 HNSCC CT 数据后，直接运行 `train.py --images /path/images --masks /path/masks`
- [ ] 是否切换到 CUDA 环境开始 GPU 完整训练？
- [ ] GPU 训练完成后，运行推理并评估 Dice / HD95？
- [ ] 是否需要调整 dataset ID、数据集名称或配置？

---

*最后更新：2026-03-06，第 4 轮*
