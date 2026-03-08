#!/usr/bin/env python3
"""
create_mock_data.py — 生成用于流程验证的 Mock CT + Mask NIfTI 数据

生成位置：
  raw_data/images/  →  case001.nii.gz … case010.nii.gz（模拟 CT，单通道 int16）
  raw_data/masks/   →  case001.nii.gz … case010.nii.gz（二值肿瘤 mask，uint8）

特性：
  - 体素大小随机（仿真临床 CT 分辨率）
  - 图像尺寸随机（~50-80 mm 各轴）
  - 肿瘤区域为球形，叠加在 CT 背景上
  - 使用 nibabel 写出标准 NIfTI1 文件
  - 已存在的文件自动跳过（幂等）

用法：
  .venv/bin/python create_mock_data.py            # 生成 10 例（默认）
  .venv/bin/python create_mock_data.py --n 20     # 生成 20 例
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("[FAIL] nibabel 未安装。请运行：pip install nibabel")

# ─────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.resolve()
IMAGES_DIR   = BASE_DIR / "raw_data" / "images"
MASKS_DIR    = BASE_DIR / "raw_data" / "masks"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# CLI 参数
# ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="生成 Mock CT/Mask NIfTI 数据")
parser.add_argument("--n", type=int, default=10, help="生成样本数量（默认 10）")
parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
parser.add_argument("--force", action="store_true", help="强制覆盖已有文件")
args = parser.parse_args()


def make_mock_case(case_id: int, rng: np.random.Generator) -> None:
    """生成一对 (CT图像, mask) NIfTI 文件。"""
    img_path  = IMAGES_DIR / f"case{case_id:03d}.nii.gz"
    mask_path = MASKS_DIR  / f"case{case_id:03d}.nii.gz"

    if img_path.exists() and mask_path.exists() and not args.force:
        print(f"  [SKIP] case{case_id:03d} 已存在，跳过")
        return

    # ── 体素尺寸（仿临床 CT：1–2 mm 面内，1.5–5 mm 层厚）
    voxel_size = np.array([
        rng.uniform(0.9, 2.0),   # x (mm)
        rng.uniform(0.9, 2.0),   # y (mm)
        rng.uniform(1.5, 5.0),   # z (mm)
    ])

    # ── 体积尺寸（约 50–80 mm 各轴，对应体素数）
    shape = tuple(
        int(rng.uniform(50, 81) / voxel_size[i]) for i in range(3)
    )  # (X, Y, Z)

    # ── CT 背景：各向同性噪声（HU 值范围 -1000 ~ 400）
    ct_volume = rng.integers(-1000, 400, size=shape, dtype=np.int16)

    # ── 球形肿瘤区域
    cx, cy, cz = [s // 2 + rng.integers(-s // 6, s // 6 + 1) for s in shape]
    radius = rng.uniform(4.0, 12.0)  # mm
    xi, yi, zi = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = np.sqrt(
        ((xi - cx) * voxel_size[0]) ** 2 +
        ((yi - cy) * voxel_size[1]) ** 2 +
        ((zi - cz) * voxel_size[2]) ** 2
    )
    tumor_mask = (dist <= radius).astype(np.uint8)

    # ── 肿瘤区域叠加较亮 HU（50–150 HU）
    ct_volume[tumor_mask == 1] = rng.integers(50, 150, size=int(tumor_mask.sum()), dtype=np.int16)

    # ── 仿射矩阵（RAS+ 方向，体素间距写入对角线）
    affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])

    # ── 写出 NIfTI
    nib.save(nib.Nifti1Image(ct_volume, affine), img_path)
    nib.save(nib.Nifti1Image(tumor_mask, affine), mask_path)
    print(
        f"  [OK] case{case_id:03d}  shape={shape}  "
        f"voxel={voxel_size.round(2).tolist()}  "
        f"tumor_voxels={tumor_mask.sum()}"
    )


# ─────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n生成 {args.n} 例 Mock CT 数据（seed={args.seed}）")
    print(f"  images → {IMAGES_DIR}")
    print(f"  masks  → {MASKS_DIR}\n")

    rng = np.random.default_rng(args.seed)
    for i in range(1, args.n + 1):
        make_mock_case(i, rng)

    img_count  = len(list(IMAGES_DIR.glob("*.nii.gz")))
    mask_count = len(list(MASKS_DIR.glob("*.nii.gz")))
    print(f"\n完成：images={img_count} 个，masks={mask_count} 个")


if __name__ == "__main__":
    main()
