# GeoMF

Implementation of [GeoMF: A Geographic Context-Aware Multimodal Framework for Fine-Grained Ship Classification](https://doi.org/10.1109/JSTARS.2026.3735688).

| Paper name | Code |
| --- | --- |
| GeoMF with the VMamba backbone | `GeoMFVMamba` in `VMamba/classification/models/vmamba.py` |
| GeoMF with the VSSD backbone | `GeoMFVSSD` in `VSSD/classification/models/mamba2.py` |
| RegionPE | `GeoShipDataset.encode_regionpe` in `classification/data/geoship.py` |
| GSCM | `GSCM` in `classification/models/geomf_modules.py` |
| HAFM and ACFU | `HAFM` and its `_apply_acfu` method in `classification/models/geomf_modules.py` |

## Training and evaluation

Use separate Linux CUDA environments for the two backbones. Install a compatible CUDA build of PyTorch before installing `requirements.txt` in the backbone directory. The VMamba configuration also uses the bundled selective-scan CUDA extension.

```bash
cd VMamba
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation ./kernels/selective_scan
torchrun --standalone --nproc_per_node=1 classification/main.py \
  --cfg classification/configs/vssm/geomf_vmamba_base.yaml \
  --data-path /path/to/GeoShip
```

```bash
cd VSSD
python -m pip install -r requirements.txt
torchrun --standalone --nproc_per_node=1 classification/main.py \
  --cfg classification/configs/vssd/geomf_vssd_base.yaml \
  --data-path /path/to/GeoShip
```

Run each command from its backbone directory after changing into it from the repository root. Pass `--pretrained /path/to/backbone.pth` to initialize from backbone weights or `--eval --resume /path/to/checkpoint.pth` to evaluate a training checkpoint.

The dataset loader expects class folders under `train/` and `test/`, per-image metadata under `info/<class>/`, and offline numeric semantic attributes in `last/train.jsonl` and `last/test.jsonl`.

The GeoShip experiments use the 12 numeric semantic fields described in the paper. Each field occupies its assigned slot in the 19-element input vector, with the other seven slots set to zero. GSCM then appends six learned graph nodes.

The original model selectors `vssm` and `vmamba2`, dataset selector `rascd`, and checkpoint parameter keys remain supported. Keep the configuration used to train a checkpoint when resuming it.

Backbone implementations are based on [VMamba](https://github.com/MzeroMiko/VMamba) and [VSSD](https://github.com/YuHengsss/VSSD).
