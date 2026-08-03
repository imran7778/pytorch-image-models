# Server execution rules

This project is intended to run on the training server.

- Keep the server dataset path in `configs/data/common_drone_rgb.yaml` unchanged:
  `/shared/data/nt_drone_dataset_all_sizes/Unified_Grayscale_Dataset`.
- Never edit paths when moving between machines. Select the checked-in `local`
  or `server` profile with `--environment`.
- Do not modify, replace, relativize, or auto-rewrite server data paths.
- Do not modify the source dataset. Materialized classifier crops must always be
  written to the configured prepared path outside `Unified_Grayscale_Dataset`.
- Every preparation run deletes and recreates only the configured prepared-crop
  directory. Never point a prepared path at a source, repository, home, or
  shared parent directory.
- The repositories `pytorch-model` and `BLEND` are independent. Runtime code in
  this repository must not import from or refer to the BLEND checkout.
- For a temporary local smoke test only, pass `--local-test`. The configured
  temporary dataset is
  `/mnt/ssd2/workstation-96/data/Unified_Grayscale_Dataset/`; this CLI override
  must never be written back to the data YAML.

Server training CLI:

```bash
cd /path/to/pytorch-model
python prepare_yolo_crop_dataset.py \
  --config configs/convnextv2_nano_yolo_crops.yaml --environment server
python train_yolo_crop_classifier.py \
  --config configs/convnextv2_nano_yolo_crops.yaml --environment server
```

Classifier export CLI:

```bash
python export_crop_classifier.py \
  --config configs/convnextv2_nano_yolo_crops.yaml --environment server
```

Both ONNX and TensorRT outputs are configured under
`/mnt/ssd2/workstation-96/export_models/convnextv2_nano/`.

Temporary local test CLI:

```bash
cd /mnt/ssd2/workstation-96/pytorch-model
python prepare_yolo_crop_dataset.py \
  --config configs/convnextv2_nano_yolo_crops.yaml \
  --environment local
python train_yolo_crop_classifier.py \
  --config configs/convnextv2_nano_yolo_crops.yaml \
  --environment local --epochs 1 --max-train-samples 25 --max-val-samples 6
```
