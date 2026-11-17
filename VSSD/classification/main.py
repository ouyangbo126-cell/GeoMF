# --------------------------------------------------------
# Modified by Yuheng Shi
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as torch_dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
from sklearn.metrics import confusion_matrix

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default='data/GeoShip', help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained', default='',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_best', action='store_true', help='After training, reload best_ckpt.pth for detailed inference instead of using latest_ckpt')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')
    parser.add_argument('--ddp', type=str, default='torch', help='distributed data parallel')
    parser.add_argument('--enable_preload', action='store_true', help='tricks to fix the bug of DataLoader')
    parser.add_argument('--enable_persistance', action='store_true', help='tricks to fix the bug of DataLoader')
    parser.add_argument('--mesa', action='store_true', help='tricks to prevent overfitting, following MLLA')
    parser.add_argument('--mesa_value', type=float, default=1.0, help='tricks to prevent overfitting, following MLLA')
    
    parser.add_argument('--confusion', type=str2bool, default=True, help='run confusion visualization')
    parser.add_argument('--tsne', type=str2bool, default=True, help='run t-SNE visualization')
    parser.add_argument('--num_hafm_blocks', '--num_fusion', dest='num_fusion', type=int, default=1, help='number of HAFM blocks')
    parser.add_argument('--use_cmf', '--use_cross_mlp', dest='use_cross_mlp', type=str2bool, default=True, help='enable CMF in ACFU')
    parser.add_argument('--use_se', type=str2bool, default=True, help='enable the optional SE layer in ACFU')
    parser.add_argument('--use_film', type=str2bool, default=True, help='use FiLM in fusion block')
    parser.add_argument('--use_ffn', type=str2bool, default=True, help='use FFN in fusion block')
    parser.add_argument('--num_cmf_layers', '--use_cross_mlp_num', dest='use_cross_mlp_num', type=int, default=1, help='number of CMF layers')
    parser.add_argument('--use_regionpe', '--use_meta', dest='use_meta', type=str2bool, default=True, help='use FFN in fusion block')
    parser.add_argument('--use_semantic', '--use_text', dest='use_text', type=str2bool, default=True, help='use FFN in fusion block')
    parser.add_argument('--use_simple_mlp_fusion', type=str2bool, default=False, help='use simple MLP fusion instead of HAFM')
    parser.add_argument('--delete_ckpt', type=str2bool, default=True, help='delete checkpoints after training to save disk space')
    parser.add_argument('--sigma', type=float, default=0.0, help='sigma for random noise on geographic coordinates (in meters), 0 means no noise')
    # RegionPE component ablation flags
    parser.add_argument('--region_pe_ebase', type=str2bool, default=True, help='Ebase: absolute coarse coordinates encoding')
    parser.add_argument('--region_pe_eoff', type=str2bool, default=True, help='Eoff: multi-scale relative offset encoding')
    parser.add_argument('--region_pe_ediff', type=str2bool, default=True, help='Ediff: offset difference encoding')
    parser.add_argument('--region_pe_edist', type=str2bool, default=True, help='Edist: haversine distance encoding')
    parser.add_argument('--region_pe_edir', type=str2bool, default=True, help='Edir: bearing direction encoding')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    if args.disable_amp:
        config.AMP_ENABLE = False

    return args, config


def main(config, args):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    # 获取实际的 geo feature 维度（根据 RegionPE 组件配置）
    if hasattr(dataset_train, 'regionpe_config'):
        _, _, _, _, _, actual_input_dim = dataset_train.regionpe_config()
        config.defrost()
        config.MODEL.LOC_INPUT_DIM = actual_input_dim
        config.freeze()
        print(f"[main] RegionPE input_dim = {actual_input_dim}")

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    if not args.mute_repeat:
        if dist.get_rank() == 0:
            logger.info(str(model))

            # 参数量计算是可靠的
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters / 1e6:.2f} M")

            # FLOPs 只能近似，或者不统计
            try:
                params, flops = model.flops_and_params(
                    img_shape=(3, 224, 224),
                    meta_dim=config.MODEL.LOC_INPUT_DIM,
                    text_dim=24,
                    verbose=False
                )
                logger.info(f"Params (from model method): {params / 1e6:.2f} M")
                logger.warning("FLOPs may be inaccurate for multi-modal/attention layers")
                logger.info(f"number of GFLOPs (approx): {flops / 1e9:.2f}")
            except Exception as e:
                logger.warning(f"FLOPs estimation failed: {e}")

    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    optimizer = build_optimizer(config, model, logger, fusion_lr_mult=5.0)
    if args.ddp == 'torch':
        model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
    else:
        raise ValueError(f"Unknown ddp type {args.ddp}")

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    steps = 0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, max_accuracy_ema, steps = load_checkpoint_ema(config, model_without_ddp, optimizer, lr_scheduler,
                                                                    loss_scaler, logger, model_ema)
        if steps + 1 == len(data_loader_train):
            config.defrost()
            config.TRAIN.START_EPOCH += 1
            config.freeze()
            steps = 0
        if config.EVAL_MODE:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
            if model_ema is not None:
                acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
                logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

            # 详细推理
            detailed_results = validate_detailed(config, data_loader_val, model_without_ddp, dataset_val, logger)
            detailed_output_path = os.path.join(config.OUTPUT, "detailed_predictions.json")
            results_to_save = {
                'predictions': detailed_results['predictions'],
                'labels': detailed_results['labels'],
                'paths': detailed_results['paths'],
                'probabilities': detailed_results['probabilities'],
                'per_class_accuracy': detailed_results['per_class_accuracy'],
                'per_class_total': [int(x) for x in detailed_results['per_class_total']],
                'per_class_correct': [int(x) for x in detailed_results['per_class_correct']],
                'overall_accuracy': detailed_results['overall_accuracy'],
                'confusion_matrix': detailed_results['confusion_matrix']
            }
            with open(detailed_output_path, 'w') as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"Detailed predictions saved to: {detailed_output_path}")
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network ema on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE and (dist.get_rank() == 0):
        logger.info(f"throughput mode ==============================")
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    writer = SummaryWriter((os.path.join(config.OUTPUT, 'tensorboard')))
    if args.enable_preload:
        # see https://github.com/triton-lang/triton/issues/3864
        for _ in data_loader_train:
            break
        for _ in data_loader_val:
            break

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler, model_ema, steps=steps, max_accuracy=max_accuracy,
                        max_accuracy_ema=max_accuracy_ema,
                        tb_writer=writer,
                        mesa=args.mesa_value if (epoch >= int(0.25 * config.TRAIN.EPOCHS) and args.mesa) else -1.0)
        steps = 0
        if dist.get_rank() == 0:
            save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                loss_scaler, logger, model_ema, max_accuracy_ema, steps=0, ckpt_name='latest_ckpt')
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # Log the accuracy to TensorBoard
        if dist.get_rank() == 0:
            writer.add_scalar('Accuracy/val', acc1, epoch)

        # Check if current accuracy is higher than the max accuracy
        if acc1 > max_accuracy:
            max_accuracy = acc1
            logger.info(f'New max accuracy: {max_accuracy:.2f}%')
            # Save the model if this is the best accuracy so far
            if dist.get_rank() == 0:
                save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                    loss_scaler, logger, model_ema, max_accuracy_ema, steps=0, ckpt_name='best_ckpt')
        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(config, data_loader_val, model_ema.ema)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_ema:.1f}%")

            # Check if current EMA accuracy is higher than the max EMA accuracy
            # Log the EMA accuracy to TensorBoard
            if dist.get_rank() == 0:
                writer.add_scalar('Accuracy_ema/val', acc1_ema, epoch)

            if acc1_ema > max_accuracy_ema:
                max_accuracy_ema = acc1_ema
                logger.info(f'New max accuracy ema: {max_accuracy_ema:.2f}%')
                # Save the model if this is the best EMA accuracy so far
                if dist.get_rank() == 0:
                    save_checkpoint_ema(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler,
                                        loss_scaler, logger, model_ema, max_accuracy_ema, steps=0,
                                        ckpt_name='best_ckpt_ema')

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    # ========== 详细推理：每张图的预测结果 + per-class accuracy ==========
    if dist.get_rank() == 0:
        logger.info("\n" + "="*70)

        # Reload best_ckpt.pth instead of using latest_ckpt for final detailed inference
        if args.eval_best:
            best_ckpt_path = os.path.join(config.OUTPUT, "best_ckpt.pth")
            logger.info(f"eval_best=True, reloading {best_ckpt_path} for detailed inference...")
            checkpoint = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            logger.info(f"Reloaded model: {msg}")
            del checkpoint
            torch.cuda.empty_cache()
            acc1_reload, acc5_reload, _ = validate(config, data_loader_val, model)
            logger.info(f"Reload best_ckpt accuracy: {acc1_reload:.1f}%")

        logger.info("Running detailed inference on test set...")
        logger.info("="*70)
        
        detailed_results = validate_detailed(config, data_loader_val, model_without_ddp, dataset_val, logger)
        
        # 保存详细结果到 JSON
        detailed_output_path = os.path.join(config.OUTPUT, "detailed_predictions.json")
        # JSON 不支持 numpy 类型，需要转换
        results_to_save = {
            'predictions': detailed_results['predictions'],
            'labels': detailed_results['labels'],
            'paths': detailed_results['paths'],
            'probabilities': detailed_results['probabilities'],
            'per_class_accuracy': detailed_results['per_class_accuracy'],
            'per_class_total': [int(x) for x in detailed_results['per_class_total']],
            'per_class_correct': [int(x) for x in detailed_results['per_class_correct']],
            'overall_accuracy': detailed_results['overall_accuracy'],
            'confusion_matrix': detailed_results['confusion_matrix']
        }
        with open(detailed_output_path, 'w') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed predictions saved to: {detailed_output_path}")
        
        # 打印 4 个 failure cases (sample predictions where pred != label)
        logger.info("\n" + "="*70)
        logger.info("Sample Failure Cases (first 10):")
        logger.info("="*70)
        failure_count = 0
        for i in range(len(detailed_results['predictions'])):
            if detailed_results['predictions'][i] != detailed_results['labels'][i]:
                pred = detailed_results['predictions'][i]
                label = detailed_results['labels'][i]
                path = detailed_results['paths'][i]
                prob = detailed_results['probabilities'][i][pred]
                logger.info(f"  [FAIL] Path: {os.path.basename(path)} | True: {label} | Pred: {pred} | Conf: {prob:.3f}")
                failure_count += 1
                if failure_count >= 10:
                    break
        logger.info(f"Total failures shown: {failure_count}")
        logger.info("="*70)

    # 训练结束后删除 checkpoints 以节省磁盘空间
    if args.delete_ckpt and dist.get_rank() == 0:
        import glob
        ckpt_dir = config.OUTPUT
        ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.pth'))
        for ckpt_file in ckpt_files:
            try:
                os.remove(ckpt_file)
                logger.info(f'Deleted checkpoint: {ckpt_file}')
            except Exception as e:
                logger.warning(f'Failed to delete {ckpt_file}: {e}')

    if args.tsne:
        logger.info("Extracting features for t-SNE...")
        feats, labels = extract_tsne_features(config, model_without_ddp, data_loader_val)

        # -----------------------------
        # TSNE 降维
        # -----------------------------
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
        feats_2d = tsne.fit_transform(feats)

        # -----------------------------
        # 颜色配置（深色，固定顺序）
        # -----------------------------
        num_classes = len(np.unique(labels))
        cmap = plt.cm.get_cmap("nipy_spectral", num_classes)

        def darken_color(color, factor=0.8):
            r, g, b, a = mcolors.to_rgba(color)
            return (r * factor, g * factor, b * factor, a)

        colors = [darken_color(cmap(i), 0.8) for i in range(num_classes)]

        # -----------------------------
        # 绘制散点图（正方形，无图例）
        # -----------------------------
        plt.figure(figsize=(6, 6))
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            plt.scatter(
                feats_2d[idx, 0],
                feats_2d[idx, 1],
                s=10,
                color=colors[i],
                alpha=0.6,  # 不透明
            )

        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        save_path = os.path.join(config.OUTPUT, "tsne.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"[t-SNE] Scatter plot saved to {save_path}")

        # -----------------------------
        # 绘制单独图例
        # -----------------------------
        legend_path = os.path.join(config.OUTPUT, "tsne_legend.png")
        plt.figure(figsize=(12, 3))
        for i in range(num_classes):
            plt.scatter([], [], color=colors[i], label=f"Class {i + 1}", s=20)
        plt.legend(ncol=10, fontsize=6, frameon=False)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(legend_path, dpi=300)
        plt.close()
        logger.info(f"[t-SNE] Legend saved to {legend_path}")

    if args.confusion:
        logger.info("Computing confusion matrix (multimodal)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        all_preds, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for idx, (samples, text, targets, metadata) in enumerate(data_loader_val):
                samples = samples.to(device, non_blocking=True)
                text = text.to(device, non_blocking=True)
                targets = targets.long().to(device, non_blocking=True)
                metadata = metadata.to(device, non_blocking=True)

                outputs = model(samples, text, metadata)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        cm = confusion_matrix(all_labels, all_preds)
        num_classes = cm.shape[0]

        # 归一化并转整数百分比
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_percent = np.nan_to_num(cm_norm * 100).round(0).astype(int)

        # 绘制高清混淆矩阵（论文白底蓝色版）
        fig, ax = plt.subplots(figsize=(7, 7), dpi=600)
        im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues", aspect="equal")
        ax.set_facecolor("white")
        ax.set_title("Confusion Matrix (%)", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        # 横纵坐标
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(range(num_classes), rotation=45, ha="center", fontsize=4)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels(range(num_classes), fontsize=4)
        ax.tick_params(axis="both", which="major", length=0)

        # 添加细网格线
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 计算动态字体大小，保证数值在格子内
        font_size = 3

        # 在每个格子中写入百分比值（0不显示）
        for i in range(num_classes):
            for j in range(num_classes):
                value = cm_percent[i, j]
                if value == 0:
                    continue
                color = "white" if value > 50 else "black"
                ax.text(
                    j, i, f"{value}",
                    ha="center", va="center",
                    color=color,
                    fontsize=font_size,
                    fontweight="bold",
                    clip_on=True
                )

        plt.tight_layout(pad=0.2)
        save_path = os.path.join(config.OUTPUT, "confusion_matrix_paper.png")
        plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close()
        logger.info(f"[Confusion Matrix] Paper-quality figure saved to {save_path}")

        # 绘制颜色条
        legend_path = os.path.join(config.OUTPUT, "confusion_colorbar_paper.png")
        fig, ax = plt.subplots(figsize=(5, 0.4), dpi=600)
        norm = plt.Normalize(vmin=0, vmax=100)
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
        cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
        cbar.set_label("Percentage (%)", fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        plt.savefig(legend_path, dpi=600, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close()
        logger.info(f"[Confusion Matrix] High-quality colorbar saved to {legend_path}")

def extract_tsne_features(config, model, data_loader, steps=0, device="cuda"):
    """
    提取用于 t-SNE 可视化的特征和标签（支持多模态融合模型）

    Args:
        config: 配置参数（可用于扩展）
        model: 已训练好的模型
        data_loader: DataLoader，返回 (samples, text, targets, metadata)
        steps: 起始步数（多卡/断点训练时可用）
        device: 运行设备，默认 'cuda'

    Returns:
        all_features (ndarray): 所有特征，shape = [N, D]
        all_labels (ndarray): 对应标签，shape = [N]
    """
    model.eval()
    all_features, all_labels = [], []
    label_counter = Counter()

    with torch.no_grad():
        for idx, (samples, text, targets, metadata) in enumerate(data_loader, start=steps):
            # 统一类型并移动到设备
            samples = samples.to(device, non_blocking=True)
            text = text.to(device, non_blocking=True)
            targets = targets.long().to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)

            # 提取 fusion 层特征
            feats = model(samples, text, metadata, return_features=True)

            # 检查 batch 是否完整
            if feats.size(0) != targets.size(0):
                raise ValueError(
                    f"Batch {idx}: 特征数量 {feats.size(0)} != 标签数量 {targets.size(0)}"
                )

            all_features.append(feats.cpu())
            all_labels.append(targets.cpu())
            label_counter.update(targets.cpu().numpy().tolist())

    # 拼接所有 batch
    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 打印类别统计
    print("每类样本数量:", dict(label_counter))
    print("总样本数:", all_features.shape[0])
    print("覆盖类别数:", len(label_counter))
    print("类别ID范围:", np.min(all_labels), "~", np.max(all_labels))

    return all_features, all_labels


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn,
                    lr_scheduler, loss_scaler, model_ema=None, model_time_warmup=50, steps=0,
                    max_accuracy=0.0, max_accuracy_ema=0.0, tb_writer=None, mesa=-1):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader) + steps
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, text, targets, metadata) in enumerate(data_loader):
        idx += steps

        torch.cuda.reset_peak_memory_stats()
        samples = samples.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        metadata = metadata.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.autocast(device_type='cuda', enabled=False):
            outputs = model(samples, text, metadata)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        grad_norm = torch.tensor(0.0)
        loss_scale_value = 1.0

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx > model_time_warmup:
            model_time.update(batch_time.val - data_time.val)

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'data time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'model time {model_time.val:.4f} ({model_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    try:
        ddp = config.MODEL.DDP
    except:
        ddp = 'torch'
    for idx, (images, text, target, metadata) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        metadata = metadata.cuda(non_blocking=True)

        # compute output
        with torch.autocast(device_type='cuda', enabled=False):
            output = model(images, text, metadata)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1, ddp=ddp)
        acc5 = reduce_tensor(acc5, ddp=ddp)
        loss = reduce_tensor(loss, ddp=ddp)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def validate_detailed(config, data_loader, model, dataset, logger):
    """
    Detailed validation that records per-image predictions with file paths.
    Returns: dict with predictions, labels, paths, and per-class accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    all_probs = []
    
    criterion = torch.nn.CrossEntropyLoss()
    loss_meter = AverageMeter()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for idx, (samples, text, targets, metadata) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        
        with torch.autocast(device_type='cuda', enabled=False):
            outputs = model(samples, text, metadata)
        
        loss = criterion(outputs, targets)
        loss_meter.update(loss.item(), targets.size(0))
        
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        # Get file paths for this batch
        batch_start = idx * data_loader.batch_size
        batch_end = batch_start + samples.size(0)
        
        for i in range(samples.size(0)):
            sample_idx = batch_start + i
            if sample_idx < len(dataset):
                img_path, _, _, _ = dataset.samples[sample_idx]
                all_paths.append(img_path)
            else:
                all_paths.append(f"unknown_{sample_idx}")
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(targets.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute per-class accuracy
    num_classes = int(max(all_labels.max(), all_preds.max())) + 1
    cm = confusion_matrix(all_labels, all_preds)
    
    per_class_correct = np.diag(cm)
    per_class_total = cm.sum(axis=1)
    per_class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if per_class_total[i] > 0:
            per_class_accuracy[i] = per_class_correct[i] / per_class_total[i] * 100
        else:
            per_class_accuracy[i] = 0.0
    
    # Log overall accuracy
    overall_acc = (all_preds == all_labels).mean() * 100
    logger.info(f'[Detailed Validate] Loss={loss_meter.avg:.4f} Acc={overall_acc:.2f}%')
    
    # Print per-class accuracy for classes 52-59
    if logger and dist.is_initialized():
        if dist.get_rank() == 0:
            logger.info("\n" + "="*60)
            logger.info("Per-Class Accuracy (Classes 52-59):")
            logger.info("="*60)
            for i in range(52, min(60, num_classes)):
                total = int(per_class_total[i])
                correct = int(per_class_correct[i])
                acc = per_class_accuracy[i]
                logger.info(f"  Class {i}: {acc:6.2f}% ({correct:4d}/{total:4d})")
            logger.info("="*60)
    elif logger:
        logger.info("\n" + "="*60)
        logger.info("Per-Class Accuracy (Classes 52-59):")
        logger.info("="*60)
        for i in range(52, min(60, num_classes)):
            total = int(per_class_total[i])
            correct = int(per_class_correct[i])
            acc = per_class_accuracy[i]
            logger.info(f"  Class {i}: {acc:6.2f}% ({correct:4d}/{total:4d})")
        logger.info("="*60)
    
    return {
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probs.tolist(),
        'paths': all_paths,
        'per_class_accuracy': per_class_accuracy.tolist(),
        'per_class_total': per_class_total.tolist(),
        'per_class_correct': per_class_correct.tolist(),
        'overall_accuracy': float(overall_acc),
        'confusion_matrix': cm.tolist()
    }


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            with torch.cuda.amp.autocast():
                model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            with torch.cuda.amp.autocast():
                model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return


if __name__ == '__main__':

    args, config = parse_option()
    stime = time.time()
    if args.ddp == 'torch':
        if torch.multiprocessing.get_start_method() != "spawn":
            # print(f"||{torch.multiprocessing.get_start_method()}||", end="")
            torch.multiprocessing.set_start_method("spawn", force=True)
        dist = torch_dist
    else:
        raise ValueError(f"Unknown ddp type {args.ddp}")

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if True:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # to make sure all the config.OUTPUT are the same
    config.defrost()
    if dist.get_rank() == 0:
        obj = [config.OUTPUT]
        # obj = [str(random.randint(0, 100))] # for test
    else:
        obj = [None]
    dist.broadcast_object_list(obj)
    dist.barrier()
    config.OUTPUT = obj[0]

    resume_file = auto_resume_helper(config.OUTPUT)
    if resume_file:
        args.mute_repeat = True
    else:
        args.mute_repeat = False

    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    if not args.mute_repeat:
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

    if args.memory_limit_rate > 0 and args.memory_limit_rate < 1:
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_rate)
        usable_memory = torch.cuda.get_device_properties(0).total_memory * args.memory_limit_rate / 1e6
        print(f"===========> GPU memory is limited to {usable_memory}MB", flush=True)

    main(config, args)
