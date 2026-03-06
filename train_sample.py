#!/usr/bin/env python3
"""
train_sample.py — nnUNet 样例训练全流程（CPU/GPU 自动适配）

步骤：
  1. 设备检测（自动选用 CPU 或 GPU，调整 epoch 数）
  2. Mock 数据检查/生成
  3. nnUNet 数据集格式化（train/test 划分 + dataset.json）
  4. plan_and_preprocess（已完成时跳过）
  5. 训练（Python API；CPU: 5 epoch 验证可行性，GPU: 1000 epoch 完整训练）
  6. 推理（测试集，通过 CLI 调用）
  7. 完成摘要

用法：
    .venv/bin/python train_sample.py
"""

import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────
# 路径 & 超参
# ─────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent.resolve()
RAW_DIR          = BASE_DIR / "nnUNet_raw"
PREPROCESSED_DIR = BASE_DIR / "nnUNet_preprocessed"
RESULTS_DIR      = BASE_DIR / "nnUNet_results"
PREDICTIONS_DIR  = BASE_DIR / "predictions"

DATASET_ID    = 100
DATASET_NAME  = "HNSCC_CT"
CONFIGURATION = "3d_fullres"
FOLD          = 0
TRAINER_NAME  = "nnUNetTrainer"
PLANS_NAME    = "nnUNetPlans"

DATASET_FOLDER = RAW_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
IMAGES_TR      = DATASET_FOLDER / "imagesTr"
IMAGES_TS      = DATASET_FOLDER / "imagesTs"
LABELS_TR      = DATASET_FOLDER / "labelsTr"
PREPROCESSED   = PREPROCESSED_DIR / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
MODEL_FOLDER   = (
    RESULTS_DIR
    / f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
    / f"{TRAINER_NAME}__{CONFIGURATION}__{PLANS_NAME}"
)

SOURCE_IMAGES = BASE_DIR / "raw_data" / "images"
SOURCE_MASKS  = BASE_DIR / "raw_data" / "masks"

# ─────────────────────────────────────────────────────────────────
# 环境变量（nnUNet 从环境变量读取路径）
# ─────────────────────────────────────────────────────────────────
os.environ["nnUNet_raw"]          = str(RAW_DIR)
os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_DIR)
os.environ["nnUNet_results"]      = str(RESULTS_DIR)

# 把 venv 的 bin 目录加入 PATH，确保 nnUNetv2_* 命令可用
VENV_BIN = BASE_DIR / ".venv" / "bin"
os.environ["PATH"] = str(VENV_BIN) + os.pathsep + os.environ.get("PATH", "")

SEP = "─" * 62


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def run_cmd(cmd: str, desc: str = "") -> None:
    """运行 shell 命令，失败时直接退出。"""
    if desc:
        print(f"  → {desc}")
    print(f"  $ {cmd}\n")
    result = subprocess.run(cmd, shell=True, env=os.environ.copy())
    if result.returncode != 0:
        sys.exit(f"\n[错误] 命令退出码 {result.returncode}，终止运行。")


def get_niis(folder: Path):
    return sorted(folder.glob("*.nii.gz")) + sorted(folder.glob("*.nii"))


# ═════════════════════════════════════════════════════════════════
# 1. 设备检测
# ═════════════════════════════════════════════════════════════════
section("1. 设备检测")

if torch.cuda.is_available():
    DEVICE       = torch.device("cuda")
    DEVICE_STR   = "cuda"
    GPU_NAME     = torch.cuda.get_device_name(0)
    MAX_EPOCHS   = 1000
    ITERS_TRAIN  = 250   # nnUNet 默认值
    ITERS_VAL    = 50
    NUM_WORKERS  = 8
    print(f"  GPU 可用: {GPU_NAME}")
    print(f"  → 完整训练模式（{MAX_EPOCHS} epoch × {ITERS_TRAIN} iter/epoch）")
else:
    DEVICE       = torch.device("cpu")
    DEVICE_STR   = "cpu"
    GPU_NAME     = None
    MAX_EPOCHS   = 5       # 仅用于验证流程（非有效模型）
    ITERS_TRAIN  = 10    # 大幅缩减，避免 CPU 等待过久
    ITERS_VAL    = 3
    NUM_WORKERS  = 2
    print("  未检测到 GPU，使用 CPU 模式")
    print(f"  → 流程验证模式（{MAX_EPOCHS} epoch × {ITERS_TRAIN} iter/epoch）")

print(f"\n  device={DEVICE}   max_epochs={MAX_EPOCHS}   iters/epoch={ITERS_TRAIN}   workers={NUM_WORKERS}")


# ═════════════════════════════════════════════════════════════════
# 2. Mock 数据检查 / 生成
# ═════════════════════════════════════════════════════════════════
section("2. Mock 数据检查")

img_files  = get_niis(SOURCE_IMAGES)
mask_files = get_niis(SOURCE_MASKS)

if not img_files:
    print("  raw_data 中无图像，正在生成 mock 数据...")
    subprocess.run(
        [sys.executable, str(BASE_DIR / "create_mock_data.py")], check=True
    )
    img_files  = get_niis(SOURCE_IMAGES)
    mask_files = get_niis(SOURCE_MASKS)
    print(f"  mock 数据生成完成：{len(img_files)} 例")
else:
    print(f"  已有 {len(img_files)} 个 mock 图像，跳过生成")

assert len(img_files) == len(mask_files), \
    f"图像({len(img_files)})与 mask({len(mask_files)})数量不一致"


# ═════════════════════════════════════════════════════════════════
# 3. 数据集格式化（imagesTr / imagesTs / labelsTr / dataset.json）
# ═════════════════════════════════════════════════════════════════
section("3. 数据集格式化")

for d in [IMAGES_TR, IMAGES_TS, LABELS_TR, PREDICTIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 固定种子 80/20 划分
random.seed(42)
indices  = list(range(len(img_files)))
random.shuffle(indices)
n_train  = math.ceil(len(indices) * 0.8)
train_idx = sorted(indices[:n_train])
test_idx  = sorted(indices[n_train:])


def _copy_case(img_path: Path, mask_path: Path, case_id: int, split: str) -> str:
    suffix    = ".nii.gz" if str(img_path).endswith(".gz") else ".nii"
    case_name = f"{DATASET_NAME}_{case_id:04d}"
    dst_dir   = IMAGES_TR if split == "train" else IMAGES_TS
    dst_img   = dst_dir / f"{case_name}_0000{suffix}"
    dst_mask  = LABELS_TR / f"{case_name}{suffix}" if split == "train" else None
    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)
    if dst_mask and not dst_mask.exists():
        shutil.copy2(mask_path, dst_mask)
    return case_name


training_cases = []
for cid, idx in enumerate(train_idx, 1):
    training_cases.append(_copy_case(img_files[idx], mask_files[idx], cid, "train"))
for cid, idx in enumerate(test_idx, n_train + 1):
    _copy_case(img_files[idx], mask_files[idx], cid, "test")

ds_json_path = DATASET_FOLDER / "dataset.json"
ds_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "tumor": 1},
    "numTraining": len(training_cases),
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO",
}
with open(ds_json_path, "w") as f:
    json.dump(ds_json, f, indent=4)

print(f"  训练集: {len(train_idx)} 例")
print(f"  测试集: {len(test_idx)} 例")
print(f"  dataset.json → {ds_json_path}")


# ═════════════════════════════════════════════════════════════════
# 4. plan_and_preprocess（已完成则跳过）
# ═════════════════════════════════════════════════════════════════
section("4. plan_and_preprocess")

preproc_config_dir = PREPROCESSED / f"nnUNetPlans_{CONFIGURATION}"
existing_b2nd = (
    list(preproc_config_dir.glob("*.b2nd"))
    if preproc_config_dir.exists()
    else []
)
# 每个训练样本对应 2 个 .b2nd（data + seg）
preproc_done = (
    (PREPROCESSED / "nnUNetPlans.json").exists()
    and len(existing_b2nd) >= n_train * 2
)

if preproc_done:
    print(f"  已检测到预处理文件（{len(existing_b2nd)} 个 .b2nd），跳过")
else:
    run_cmd(
        f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} "
        f"--verify_dataset_integrity -np {NUM_WORKERS}",
        "运行 plan_and_preprocess",
    )


# ═════════════════════════════════════════════════════════════════
# 5. 训练（Python API，控制 epoch 数；CPU/GPU 均可）
# ═════════════════════════════════════════════════════════════════
section(f"5. 训练  [device={DEVICE}, max_epochs={MAX_EPOCHS}]")

checkpoint_final = MODEL_FOLDER / f"fold_{FOLD}" / "checkpoint_final.pth"

if checkpoint_final.exists():
    print(f"  已检测到 checkpoint_final.pth，跳过训练")
else:
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    plans      = load_json(str(PREPROCESSED / "nnUNetPlans.json"))
    dataset_json = load_json(str(ds_json_path))

    # 在实例化前覆盖类属性，控制训练轮数
    nnUNetTrainer.max_num_epochs = MAX_EPOCHS

    print(f"  初始化 nnUNetTrainer（configuration={CONFIGURATION}, fold={FOLD}）")
    trainer = nnUNetTrainer(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        device=DEVICE,
    )
    trainer.initialize()

    print(f"\n  开始训练（共 {MAX_EPOCHS} epoch）...\n")
    trainer.run_training()
    print("\n  训练完成")


# ═════════════════════════════════════════════════════════════════
# 6. 推理（测试集；通过 CLI 调用）
# ═════════════════════════════════════════════════════════════════
section("6. 推理（测试集）")

ckpt_best  = MODEL_FOLDER / f"fold_{FOLD}" / "checkpoint_best.pth"
ckpt_final = MODEL_FOLDER / f"fold_{FOLD}" / "checkpoint_final.pth"

if ckpt_best.exists():
    ckpt_name = "checkpoint_best.pth"
elif ckpt_final.exists():
    ckpt_name = "checkpoint_final.pth"
else:
    print("  未找到可用 checkpoint，跳过推理")
    ckpt_name = None

if ckpt_name:
    # nnUNetv2_predict 通过 -device 参数指定设备
    run_cmd(
        f'nnUNetv2_predict '
        f'-i "{IMAGES_TS}" '
        f'-o "{PREDICTIONS_DIR}" '
        f'-d {DATASET_ID} '
        f'-c {CONFIGURATION} '
        f'-tr {TRAINER_NAME} '
        f'-chk {ckpt_name} '
        f'-f {FOLD} '
        f'-device {DEVICE_STR} '
        f'-npp {max(1, NUM_WORKERS // 2)} '
        f'-nps {max(1, NUM_WORKERS // 2)}',
        f"推理测试集（checkpoint={ckpt_name}）",
    )
    pred_files = list(PREDICTIONS_DIR.glob("*.nii*"))
    print(f"\n  生成预测文件: {len(pred_files)} 个")
    for p in pred_files:
        print(f"    {p.name}")


# ═════════════════════════════════════════════════════════════════
# 7. 完成摘要
# ═════════════════════════════════════════════════════════════════
section("7. 完成摘要")

gpu_info = f"  GPU          : {GPU_NAME}\n" if GPU_NAME else ""
print(f"""
  设备         : {DEVICE}
{gpu_info}  配置         : {CONFIGURATION},  fold={FOLD}
  训练 epoch   : {MAX_EPOCHS}
  模型目录     : {MODEL_FOLDER}
  推理输出     : {PREDICTIONS_DIR}

  说明：
    · CPU 模式的 {MAX_EPOCHS} epoch 仅验证流程完整性，不产生有效分割模型
    · 切换到 CUDA 环境后重新运行，脚本会自动以 1000 epoch 完整训练
    · 各步骤有完成检测：已有预处理/checkpoint 时自动跳过，可增量运行
""")
