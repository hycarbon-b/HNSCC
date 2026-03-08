#!/usr/bin/env python3
"""
train.py  —  HNSCC nnUNet 训练脚本（CPU / GPU 兼容）

用法：
  .venv/bin/python train.py --images /path/to/images --masks /path/to/masks
  .venv/bin/python train.py --images /path/to/images --masks /path/to/masks \\
      --dataset-id 101 --dataset-name MY_DATASET --fold 0 --epochs 500

必需参数：
  --images   包含 CT NIfTI 文件（*.nii.gz）的目录
  --masks    包含对应分割 mask NIfTI 文件（*.nii.gz）的目录
             文件名须与 images 目录一一对应（排序后匹配）

可选参数：
  --dataset-id    数据集 ID（默认 100）
  --dataset-name  数据集名称（默认 HNSCC_CT）
  --fold          交叉验证 fold（默认 0）
  --train-split   训练集比例（默认 0.8）
  --channel       CT 通道名称，写入 dataset.json（默认 CT）
  --label         前景标签名，写入 dataset.json（默认 tumor）
  --epochs        覆盖训练 epoch 数（默认：CPU=5，GPU=1000）
  --config        nnUNet 配置（默认：CPU=2d，GPU=3d_fullres）
  --force-preprocess  强制重新运行 plan_and_preprocess
"""

# ── 必须在导入 nnunetv2 之前设置环境变量 ──────────────────────────
import os, sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
os.environ['nnUNet_raw']          = str(BASE_DIR / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(BASE_DIR / 'nnUNet_preprocessed')
os.environ['nnUNet_results']      = str(BASE_DIR / 'nnUNet_results')

# ── 标准库导入 ──────────────────────────────────────────────────
import argparse, json, shutil, math, random, subprocess

# ── 第三方库导入 ────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print('[FAIL] torch 未安装。请运行：uv pip install torch')
    sys.exit(1)

# ── CLI 参数解析 ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='nnUNet v2 训练脚本（接受真实 NIfTI 数据集路径）',
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--images',  required=True,
                    help='CT NIfTI 文件目录（*.nii.gz）')
parser.add_argument('--masks',   required=True,
                    help='对应 mask NIfTI 文件目录（*.nii.gz，与 images 排序匹配）')
parser.add_argument('--dataset-id',   type=int, default=100,
                    help='数据集 ID（默认 100）')
parser.add_argument('--dataset-name', default='HNSCC_CT',
                    help='数据集名称（默认 HNSCC_CT）')
parser.add_argument('--fold',    type=int, default=0,
                    help='交叉验证 fold（默认 0）')
parser.add_argument('--train-split', type=float, default=0.8,
                    help='训练集比例（默认 0.8）')
parser.add_argument('--channel', default='CT',
                    help='CT 通道名称，写入 dataset.json（默认 CT）')
parser.add_argument('--label',   default='tumor',
                    help='前景标签名，写入 dataset.json（默认 tumor）')
parser.add_argument('--epochs',  type=int, default=None,
                    help='覆盖训练 epoch 数')
parser.add_argument('--config',  default=None,
                    help='nnUNet 配置：2d / 3d_fullres（默认由设备自动选择）')
parser.add_argument('--force-preprocess', action='store_true',
                    help='强制重新运行 plan_and_preprocess')
# WandB 参数
parser.add_argument('--wandb', action='store_true',
                    help='启用 Weights & Biases 训练监控')
parser.add_argument('--wandb-project', default='hnscc-nnunet',
                    help='WandB project 名称（默认 hnscc-nnunet）')
parser.add_argument('--wandb-name', default=None,
                    help='WandB run 名称（默认自动生成）')
parser.add_argument('--wandb-tags', nargs='*', default=None,
                    help='WandB 标签列表，用空格分隔')
args = parser.parse_args()

# ── CPU / GPU 检测与训练参数 ────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IS_GPU = DEVICE.type == 'cuda'

if args.config:
    CONFIGURATION = args.config
elif IS_GPU:
    CONFIGURATION = '3d_fullres'
else:
    CONFIGURATION = '2d'

if args.epochs is not None:
    NUM_EPOCHS = args.epochs
elif IS_GPU:
    NUM_EPOCHS = 1000
else:
    NUM_EPOCHS = 5

NUM_ITER_TRAIN = 250 if IS_GPU else 20
NUM_ITER_VAL   = 50  if IS_GPU else 5

DATASET_ID   = args.dataset_id
DATASET_NAME = args.dataset_name
FOLD         = args.fold

# ── 路径 ────────────────────────────────────────────────────────
IMAGES_SRC   = Path(args.images)
MASKS_SRC    = Path(args.masks)
DATASET_DIR  = BASE_DIR / 'nnUNet_raw' / f'Dataset{DATASET_ID:03d}_{DATASET_NAME}'
IMAGES_TR    = DATASET_DIR / 'imagesTr'
IMAGES_TS    = DATASET_DIR / 'imagesTs'
LABELS_TR    = DATASET_DIR / 'labelsTr'
PREPROCESSED = BASE_DIR / 'nnUNet_preprocessed' / f'Dataset{DATASET_ID:03d}_{DATASET_NAME}'
VENV_BIN     = BASE_DIR / '.venv' / 'bin'

SEP = '─' * 60


def section(title: str):
    print(f'\n{SEP}\n{title}\n{SEP}')


def info(msg: str):
    print(f'  {msg}')


# ════════════════════════════════════════════════════════════════
# Step 1 —— 验证输入路径与文件配对
# ════════════════════════════════════════════════════════════════
section('Step 1 — 验证输入数据')

if not IMAGES_SRC.is_dir():
    print(f'[FAIL] --images 目录不存在: {IMAGES_SRC}')
    sys.exit(1)
if not MASKS_SRC.is_dir():
    print(f'[FAIL] --masks 目录不存在: {MASKS_SRC}')
    sys.exit(1)

img_files  = sorted(IMAGES_SRC.glob('*.nii.gz'))
mask_files = sorted(MASKS_SRC.glob('*.nii.gz'))

if not img_files:
    print(f'[FAIL] --images 目录中未找到 *.nii.gz 文件: {IMAGES_SRC}')
    sys.exit(1)
if not mask_files:
    print(f'[FAIL] --masks 目录中未找到 *.nii.gz 文件: {MASKS_SRC}')
    sys.exit(1)
if len(img_files) != len(mask_files):
    print(f'[FAIL] 图像数量（{len(img_files)}）与 mask 数量（{len(mask_files)}）不一致')
    sys.exit(1)

for img, mask in zip(img_files, mask_files):
    if img.name != mask.name:
        print(f'[FAIL] 文件名不匹配: {img.name} vs {mask.name}')
        print('       请确保 images/ 和 masks/ 目录中文件名完全相同（排序后一一对应）')
        sys.exit(1)

info(f'图像路径 : {IMAGES_SRC}')
info(f'Mask 路径: {MASKS_SRC}')
info(f'共 {len(img_files)} 个样本，文件名配对验证通过。')


# ════════════════════════════════════════════════════════════════
# Step 2 —— 格式化 nnUNet 数据集
# ════════════════════════════════════════════════════════════════
section('Step 2 — nnUNet 数据集格式化')

for d in [IMAGES_TR, IMAGES_TS, LABELS_TR]:
    d.mkdir(parents=True, exist_ok=True)

# 若 imagesTr/ 已有文件则跳过（避免重复复制）
if list(IMAGES_TR.glob('*.nii*')):
    n_tr = len(list(IMAGES_TR.glob('*.nii*')))
    n_ts = len(list(IMAGES_TS.glob('*.nii*')))
    info(f'已格式化：{n_tr} 训练 / {n_ts} 测试，跳过复制。')
else:
    split_ratio = max(0.0, min(1.0, args.train_split))
    n_train = max(1, math.ceil(len(img_files) * split_ratio))

    random.seed(42)
    idxs = list(range(len(img_files)))
    random.shuffle(idxs)
    tr_idx = sorted(idxs[:n_train])
    ts_idx = sorted(idxs[n_train:])

    info(f'划分：{len(tr_idx)} 训练 / {len(ts_idx)} 测试（split={split_ratio:.0%}）')

    training_cases = []
    for cid, idx in enumerate(tr_idx, start=1):
        base     = f'{DATASET_NAME}_{cid:04d}'
        shutil.copy2(img_files[idx],  IMAGES_TR / f'{base}_0000.nii.gz')
        shutil.copy2(mask_files[idx], LABELS_TR / f'{base}.nii.gz')
        training_cases.append(base)
        info(f'  [TR] {img_files[idx].name} → {base}')

    for cid, idx in enumerate(ts_idx, start=n_train + 1):
        base = f'{DATASET_NAME}_{cid:04d}'
        shutil.copy2(img_files[idx], IMAGES_TS / f'{base}_0000.nii.gz')
        info(f'  [TS] {img_files[idx].name} → {base}')

    ds_json = {
        'channel_names': {'0': args.channel},
        'labels':        {'background': 0, args.label: 1},
        'numTraining':   len(training_cases),
        'file_ending':   '.nii.gz',
        'overwrite_image_reader_writer': 'SimpleITKIO',
    }
    with open(DATASET_DIR / 'dataset.json', 'w') as f:
        json.dump(ds_json, f, indent=4)

    info(f'dataset.json 已写入：channel={args.channel}, label={args.label}')


# ════════════════════════════════════════════════════════════════
# Step 3 —— plan_and_preprocess
# ════════════════════════════════════════════════════════════════
section('Step 3 — plan_and_preprocess')

plans_file  = PREPROCESSED / 'nnUNetPlans.json'
data_dir_2d = PREPROCESSED / 'nnUNetPlans_2d'
data_dir_3d = PREPROCESSED / 'nnUNetPlans_3d_fullres'

already_done = (
    not args.force_preprocess and
    plans_file.exists() and
    data_dir_2d.exists() and any(data_dir_2d.glob('*.b2nd')) and
    data_dir_3d.exists() and any(data_dir_3d.glob('*.b2nd'))
)

if already_done:
    info('检测到预处理文件已存在，跳过（使用 --force-preprocess 强制重新运行）。')
else:
    if args.force_preprocess:
        info('--force-preprocess：强制重新运行。')
    info('开始 plan_and_preprocess…')
    cmd = (
        f'nnUNetv2_plan_and_preprocess -d {DATASET_ID} '
        f'--verify_dataset_integrity -np 2'
    )
    env = os.environ.copy()
    env['PATH'] = str(VENV_BIN) + os.pathsep + env.get('PATH', '')
    result = subprocess.run(
        cmd, shell=True, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f'[FAIL] plan_and_preprocess 失败（exit {result.returncode}）')
        sys.exit(1)
    info('plan_and_preprocess 完成。')


# ════════════════════════════════════════════════════════════════
# Step 4 —— 训练
# ════════════════════════════════════════════════════════════════
section('Step 4 — 训练')

info(f'设备    : {DEVICE}  ({"GPU ✓" if IS_GPU else "CPU — smoke test 模式"})')
info(f'配置    : {CONFIGURATION}')
info(f'Fold    : {FOLD}')
info(f'Epochs  : {NUM_EPOCHS}')
info(f'Iter/ep : {NUM_ITER_TRAIN} (train)  /  {NUM_ITER_VAL} (val)')

try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from batchgenerators.utilities.file_and_folder_operations import load_json
except ImportError as e:
    print(f'[FAIL] nnunetv2 导入失败: {e}')
    sys.exit(1)


# ── WandB Trainer 子类（仅在 --wandb 时使用）────────────────────
class WandbNnUNetTrainer(nnUNetTrainer):
    """在 nnUNetTrainer 基础上，每个 epoch 结束后将指标上报至 WandB。"""

    def __init__(self, *args,
                 wandb_project: str = 'hnscc-nnunet',
                 wandb_name: str | None = None,
                 wandb_tags: list | None = None,
                 wandb_config: dict | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._wb_project = wandb_project
        self._wb_name    = wandb_name
        self._wb_tags    = wandb_tags
        self._wb_config  = wandb_config or {}
        self._wandb      = None   # 延迟导入

    def initialize(self):
        super().initialize()
        try:
            import wandb as _wandb
        except ImportError:
            print('[WARN] wandb 未安装（pip install wandb），WandB 监控已禁用')
            return
        self._wandb = _wandb
        self._wandb.init(
            project=self._wb_project,
            name=self._wb_name,
            tags=self._wb_tags,
            config={
                'configuration':            self.configuration_name,
                'fold':                     self.fold,
                'num_epochs':               self.num_epochs,
                'num_iterations_per_epoch': self.num_iterations_per_epoch,
                'device':                   str(self.device),
                'dataset_id':               DATASET_ID,
                'dataset_name':             DATASET_NAME,
                **self._wb_config,
            },
            resume='allow',
        )
        print(f'  [WandB] run: {self._wandb.run.url}')

    def on_epoch_end(self):
        """调用父类逻辑后，将当前 epoch 的指标推送至 WandB。"""
        super().on_epoch_end()   # 注意：此调用后 self.current_epoch 已 +1
        if self._wandb is None or self._wandb.run is None:
            return
        logs = self.logger.my_fantastic_logging
        metrics: dict = {'epoch': self.current_epoch - 1}
        # ── nnUNet logger 存储各 epoch 指标为列表，取最后一项 ──
        for key, wb_key in [
            ('train_losses',  'train_loss'),
            ('val_losses',    'val_loss'),
            ('mean_fg_dice',  'mean_fg_dice'),
            ('ema_fg_dice',   'ema_fg_dice'),
            ('lrs',           'lr'),
        ]:
            if logs.get(key):
                metrics[wb_key] = logs[key][-1]
        self._wandb.log(metrics)

    def on_train_end(self):
        super().on_train_end()
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()
            print('  [WandB] run 已结束')


# ── 选择 Trainer 类 ────────────────────────────────────────────
USE_WANDB = args.wandb
TrainerClass = WandbNnUNetTrainer if USE_WANDB else nnUNetTrainer

if USE_WANDB:
    info(f'WandB   : 已启用  project={args.wandb_project}'
         + (f'  name={args.wandb_name}' if args.wandb_name else ''))
else:
    info('WandB   : 未启用（加 --wandb 开启）')

plans        = load_json(str(plans_file))
dataset_json = load_json(str(DATASET_DIR / 'dataset.json'))

if USE_WANDB:
    trainer = WandbNnUNetTrainer(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        device=DEVICE,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_config={
            'epochs':      NUM_EPOCHS,
            'iter_train':  NUM_ITER_TRAIN,
            'iter_val':    NUM_ITER_VAL,
        },
    )
else:
    trainer = nnUNetTrainer(
        plans=plans,
        configuration=CONFIGURATION,
        fold=FOLD,
        dataset_json=dataset_json,
        device=DEVICE,
    )

# 覆盖训练规模（必须在 initialize() 之前，LR 调度器会用到 num_epochs）
trainer.num_epochs                   = NUM_EPOCHS
trainer.num_iterations_per_epoch     = NUM_ITER_TRAIN
trainer.num_val_iterations_per_epoch = NUM_ITER_VAL

trainer.initialize()

info(f'\n  输出目录: {trainer.output_folder}')
info('  开始训练…\n')

trainer.run_training()

section('全流程完成')
print(f"""
  图像来源  : {IMAGES_SRC}
  Mask 来源 : {MASKS_SRC}
  nnUNet 格式: {DATASET_DIR}
  训练结果  : {trainer.output_folder}

  配置 : {CONFIGURATION}  |  设备 : {DEVICE}  |  Epochs : {NUM_EPOCHS}
""")
