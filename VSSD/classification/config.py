# --------------------------------------------------------
# Modified by $@#Anonymous#@$
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = 'data/GeoShip'
# Dataset name
_C.DATA.DATASET = 'geoship'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 0

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

#persistent workers
_C.DATA.PERSISTENT_WORKERS = False
# Sigma for random noise on geographic coordinates (meters), 0 means no noise
_C.DATA.SIGMA = 0.0
# RegionPE component ablation settings
_C.DATA.REGION_PE_EBASE = True   # Ebase: absolute coarse coordinates (sin/cos encoding)
_C.DATA.REGION_PE_EOFF = True    # Eoff: multi-scale relative offsets
_C.DATA.REGION_PE_EDIFF = True   # Ediff: offset difference encoding
_C.DATA.REGION_PE_EDIST = True    # Edist: haversine distance encoding
_C.DATA.REGION_PE_EDIR = True     # Edir: bearing direction encoding
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'geomf_vssd'
# Model name
_C.MODEL.NAME = 'geomf_vssd'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 60
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rates
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1
#for ddp platform
_C.MODEL.DDP = 'torch'
# Location input dimension (for RegionPE ablation)
_C.MODEL.LOC_INPUT_DIM = 24

# MMpretrain models for test
_C.MODEL.MMCKPT = False

# VMAMBA2 parameters
_C.MODEL.VMAMBA2 = CN()
_C.MODEL.VMAMBA2.PATCH_SIZE = 4
_C.MODEL.VMAMBA2.IN_CHANS = 3
_C.MODEL.VMAMBA2.NUM_HEADS = [2, 4, 8, 16]
_C.MODEL.VMAMBA2.DEPTHS = [2, 4, 8, 4]
_C.MODEL.VMAMBA2.EMBED_DIM = 64
_C.MODEL.VMAMBA2.MLP_RATIO = 4.0
_C.MODEL.VMAMBA2.DROP_RATE = 0.0
_C.MODEL.VMAMBA2.DROP_PATH_RATE = 0.2
_C.MODEL.VMAMBA2.IMAGE_SIZE = 224
_C.MODEL.VMAMBA2.SIMPLE_DOWNSAMPLE = False
_C.MODEL.VMAMBA2.SIMPLE_PATCH_EMBED = False
_C.MODEL.VMAMBA2.SSD_EXPANSION = 2
_C.MODEL.VMAMBA2.SSD_NGROUPS = 1
_C.MODEL.VMAMBA2.SSD_CHUNK_SIZE = 256
_C.MODEL.VMAMBA2.LINEAR_ATTN_DUALITY = True
_C.MODEL.VMAMBA2.ATTN_TYPES = ['mamba2', 'mamba2', 'mamba2', 'standard']
_C.MODEL.VMAMBA2.LEPE = False
_C.MODEL.VMAMBA2.BIDIRECTION = False
_C.MODEL.VMAMBA2.D_STATE = 64
_C.MODEL.VMAMBA2.SSD_POSITIVE_DA = True
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 3e-3
_C.TRAIN.WARMUP_LR = 5e-5
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = False
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Test traincost only, overwritten by command line argument
_C.TRAINCOST_MODE = False
# for acceleration
_C.FUSED_LAYERNORM = False

# for abotion
_C.TSNE = True   # 默认值
_C.FUSION = CN()                 # 先定义 CN 节点
_C.FUSION.NUM_FUSION = 1       # 默认融合层数
_C.FUSION.USE_CROSS_MLP = True # 是否使用 Cross MLP
_C.FUSION.USE_SE = True        # 是否使用 SE 模块
_C.FUSION.USE_FILM = True      # 是否使用 FiLM
_C.FUSION.USE_FFN = True       # 是否使用 FFN
_C.FUSION.USE_CROSS_MLP_NUM = 1       # 是否使用 FFN
_C.FUSION.USE_META = True      # 是否使用 FiLM
_C.FUSION.USE_TEXT = True       # 是否使用 FFN
_C.FUSION.USE_SIMPLE_MLP_FUSION = False  # 使用简单 MLP Fusion 替代 HAFM

def _recursive_update_config_from_file(config, cfg_file):
    # 不调用 freeze()，只做递归合并
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _recursive_update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)


def update_config(config, args):
    config.defrost()  # 解锁，准备修改

    _recursive_update_config_from_file(config, args.cfg)  # 递归合并 config 文件

    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        return hasattr(args, name) and bool(getattr(args, name))

    # 下面覆盖命令行参数到配置
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('traincost'):
        config.TRAINCOST_MODE = True
    if _check_args('enable_persistance'):
        config.DATA.PERSISTENT_WORKERS = True
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim
    if _check_args('ddp'):
        config.MODEL.DDP = args.ddp
    if _check_args('tsne'):
        config.defrost()
        setattr(config, "TSNE", args.tsne)
    if hasattr(args, 'num_fusion'):
        config.FUSION.NUM_FUSION = args.num_fusion
    if hasattr(args, 'use_cross_mlp_num'):
        config.FUSION.USE_CROSS_MLP_NUM = args.use_cross_mlp_num
    if hasattr(args, 'use_cross_mlp'):
        config.FUSION.USE_CROSS_MLP = args.use_cross_mlp
    if hasattr(args, 'use_se'):
        config.FUSION.USE_SE = args.use_se
    if hasattr(args, 'use_film'):
        config.FUSION.USE_FILM = args.use_film
    if hasattr(args, 'use_ffn'):
        config.FUSION.USE_FFN = args.use_ffn
    if hasattr(args, 'use_meta'):
        config.FUSION.USE_META = args.use_meta
    if hasattr(args, 'use_text'):
        config.FUSION.USE_TEXT = args.use_text
    if hasattr(args, 'use_simple_mlp_fusion'):
        config.FUSION.USE_SIMPLE_MLP_FUSION = args.use_simple_mlp_fusion
    if hasattr(args, 'sigma'):
        config.DATA.SIGMA = args.sigma
    # RegionPE component ablation flags
    if hasattr(args, 'region_pe_ebase'):
        config.DATA.REGION_PE_EBASE = args.region_pe_ebase
    if hasattr(args, 'region_pe_eoff'):
        config.DATA.REGION_PE_EOFF = args.region_pe_eoff
    if hasattr(args, 'region_pe_ediff'):
        config.DATA.REGION_PE_EDIFF = args.region_pe_ediff
    if hasattr(args, 'region_pe_edist'):
        config.DATA.REGION_PE_EDIST = args.region_pe_edist
    if hasattr(args, 'region_pe_edir'):
        config.DATA.REGION_PE_EDIR = args.region_pe_edir
    # 拼接 output 目录
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()  # 最后锁定配置，不允许修改


def get_config(args):
    config = _C.clone()  # 复制默认配置，避免修改全局
    update_config(config, args)
    return config
