# Brazilian Army Armored Vehicle Identification Pipeline

An end-to-end pipeline for building an instance-segmentation model that recognizes armored vehicles of the Brazilian Army, and running it in real time on a Raspberry Pi. It goes from raw web images to an edge-deployed YOLOv8-seg model in five stages:

```
 1. repositorio_viaturas.py   ->  collect + deduplicate images
 2. pre_anotacao_gsam.py      ->  auto-annotate masks (GroundingDINO + SAM) -> COCO JSON
    (manual review in CVAT / Label Studio)
 3. treino_yolov8seg.py       ->  COCO -> YOLO-seg conversion, training, evaluation
 4. export_rasp.py            ->  export best.pt to NCNN for ARM CPUs (Raspberry Pi)
 5. yolo_viat_eb.py           ->  real-time webcam inference on the Raspberry Pi
```

## Target classes

| ID (COCO) | YOLO idx | Class          | Vehicle                               |
|-----------|----------|----------------|---------------------------------------|
| 1         | 0        | `VBTP_Guarani` | VBTP-MR Guarani (6x6 APC)             |
| 2         | 1        | `Cascavel_EE9` | EE-9 Cascavel (wheeled armored car)   |
| 3         | 2        | `VBC_Leopard`  | Leopard 1A5 BR (main battle tank)     |
| 4         | 3        | `Guepard_M113` | Guepard (M113 APC)                    |
| 5         | 4        | `Lince_LMV`    | Lince (Iveco LMV 4x4 light armored)   |

## Repository layout

```
.
├── repositorio_viaturas.py
├── pre_anotacao_gsam.py
├── treino_yolov8seg.py
├── export_rasp.py
├── yolo_viat_eb.py
│
├── repositorio_viaturas/          # stage 1 output
│   ├── <Class>/imagens/*.jpg
│   └── relatorio.json
├── modelos_gsam/                  # stage 2: weights, config, GroundingDINO clone
├── anotacoes_coco/                # stage 2 output
│   ├── <Class>_coco.json
│   └── relatorio_anotacao.json
├── dataset_yolo/                  # stage 3: converted dataset
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── dataset.yaml
└── runs/segment/viaturas_eb/      # stage 3: training results
    └── weights/
        ├── best.pt, last.pt
        └── best_ncnn_model/       # stage 4 output (copy this folder to the Pi)
```

## Requirements

- Python 3.9+
- Google Chrome installed (stage 1)
- Git installed and on `PATH` (stage 2 clones GroundingDINO)
- An NVIDIA GPU is recommended for stages 2 and 3, but all scripts run on CPU
- Stage 5: a Raspberry Pi 4 (or similar ARM board) with a USB or Pi camera and a connected display

Each script checks its own dependencies and tells you what to install if something is missing.

---

## Stage 1 — Image collection (`repositorio_viaturas.py`)

Downloads images for each vehicle using a real Chrome browser driven by Selenium (to avoid the anti-bot blocks that affect plain HTTP scraping), then removes near-duplicates with perceptual hashing.

**How it works**

1. Opens Bing Images for each of the three queries defined per vehicle and scrolls to load results. If Bing returns fewer than 5 URLs, it falls back to Google Images.
2. Downloads each image, discarding anything smaller than 200×150 px, and re-saves it as JPEG (quality 90).
3. Filters near-duplicates using pHash (`hash_size=16`); an image is kept only if its Hamming distance to every already-kept image is greater than `--threshold`.
4. Writes `relatorio.json` with downloaded / kept / removed counts per class.

**Install**

```bash
pip install selenium webdriver-manager Pillow imagehash tqdm requests
```

**Usage**

```bash
python repositorio_viaturas.py                          # 30 images/vehicle, threshold 8
python repositorio_viaturas.py --images 50              # more images per vehicle
python repositorio_viaturas.py --threshold 5            # keep more (less aggressive filtering)
python repositorio_viaturas.py --output C:/my_dataset   # custom output folder
python repositorio_viaturas.py --visible                # show the browser (debugging)
```

| Option        | Default                  | Description                                          |
|---------------|--------------------------|------------------------------------------------------|
| `--images`    | `30`                     | Images to download per vehicle                       |
| `--threshold` | `8`                      | pHash Hamming-distance threshold; lower keeps more   |
| `--output`    | `repositorio_viaturas`   | Root output folder                                   |
| `--visible`   | off                      | Run Chrome with a visible window                     |

**Notes**

- Driver setup tries, in order: Selenium Manager (Selenium ≥ 4.10), `webdriver-manager`, then a `chromedriver` on `PATH`.
- With `hash_size=16` the hash has 256 bits, so the effective threshold range is 0–256 (the `--help` text mentions 0–64, which applies to the default 8×8 hash). Values around 8–20 are a reasonable range to experiment with.
- Respect the terms of use and copyright of the sites and images you collect. Images found via search engines are for internal research/training only unless you verify their licenses.

---

## Stage 2 — Auto-annotation (`pre_anotacao_gsam.py`)

Generates segmentation masks automatically with **Grounded-SAM**: GroundingDINO detects the vehicle from a text prompt, and SAM turns each detected box into a mask. Output is COCO JSON, compatible with CVAT and Label Studio for manual review.

### One-time setup

```bash
python pre_anotacao_gsam.py --instalar
```

This installs PyTorch (CUDA 11.8 build), `transformers==4.35.2`, `timm==0.9.12`, `supervision==0.19.0`, Segment Anything and other dependencies; clones GroundingDINO (registered via a `.pth` file, so **no C++ compiler is needed on Windows**); and downloads the model weights into `modelos_gsam/`.

It also detects your hardware and asks which SAM profile to download:

| Profile  | SAM model | Size    | Suggested hardware           |
|----------|-----------|---------|------------------------------|
| `leve`   | ViT-B     | ~375 MB | CPU / weak PC                |
| `medio`  | ViT-L     | ~1.2 GB | GPU with 6–8 GB VRAM         |
| `pesado` | ViT-H     | ~2.4 GB | GPU with 10+ GB VRAM         |

If you hit a `transformers` / GroundingDINO incompatibility or a `groundingdino` import error, run:

```bash
python pre_anotacao_gsam.py --corrigir
```

### Running the annotation

```bash
# Annotate every vehicle in the repository from stage 1
python pre_anotacao_gsam.py --repo repositorio_viaturas

# Annotate a single folder
python pre_anotacao_gsam.py --pasta repositorio_viaturas/VBTP_Guarani/imagens --viatura VBTP_Guarani

# Higher/lower detection confidence, plus a visual preview grid
python pre_anotacao_gsam.py --repo repositorio_viaturas --confianca 0.40 --preview
```

| Option        | Default          | Description                                                       |
|---------------|------------------|-------------------------------------------------------------------|
| `--instalar`  | —                | Install dependencies and download weights                         |
| `--corrigir`  | —                | Fix dependency/module problems without a full reinstall           |
| `--repo`      | —                | Repository root (processes all five vehicle folders)              |
| `--pasta`     | —                | A single image folder                                             |
| `--viatura`   | folder's parent  | Class name to use with `--pasta`                                  |
| `--confianca` | `0.35`           | GroundingDINO box threshold (text threshold = 0.7 × this value)   |
| `--pesos`     | `modelos_gsam`   | Folder with model weights                                         |
| `--saida`     | `anotacoes_coco` | Output folder for COCO JSONs                                      |
| `--preview`   | off              | Save a 3×2 overlay grid (`<Class>_coco_preview.jpg`)              |
| `--cpu`       | off              | Force CPU even if a GPU is available                              |

At load time the script automatically picks the best SAM checkpoint found in `modelos_gsam/` (ViT-H > ViT-L > ViT-B on GPU; the reverse order on CPU).

### Output

- `anotacoes_coco/<Class>_coco.json` — one COCO file per vehicle, with polygon segmentations, bounding boxes, areas, the SAM score, and `"auto_labeled": true` on every annotation.
- `anotacoes_coco/relatorio_anotacao.json` — summary: images processed, annotated, and with no detection.

### Important caveats

- **Every detection in a folder is assigned that folder's class.** Backgrounds, other vehicles, or false positives are labeled as the class too, so manual review is essential.
- Prompts are generic per vehicle type (e.g. `"military main battle tank tracks turret"`) rather than model-specific, since the text model cannot reliably tell a Guarani from other 6x6 APCs. Class identity comes from the folder, not from the detector.
- Images with no detection are listed in the summary and should be annotated by hand.

### Recommended review step

Import the JSONs into **CVAT** or **Label Studio**, correct/remove masks, and export back to COCO. Keep the original file names (`<Class>_coco.json`) and image file names, because stage 3 depends on them.

---

## Stage 3 — Training (`treino_yolov8seg.py`)

Converts the reviewed COCO annotations to YOLO-seg format, splits train/val, and trains a YOLOv8-seg model with parameters tuned to your hardware.

**Install**

```bash
pip install ultralytics pycocotools Pillow tqdm pyyaml
```

**Expected inputs**

- `--coco`: folder containing `<Class>_coco.json` for each class
- `--imagens`: repository root containing `<Class>/imagens/` folders (as produced by stage 1)

**Usage**

```bash
# Full flow: prepare dataset + train + evaluate
python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas

# Only build the dataset
python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas --so-preparar

# Force a hardware profile and epoch count
python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas --perfil gpu_alta --epochs 200

# Resume an interrupted run
python treino_yolov8seg.py --retomar runs/segment/viaturas_eb/weights/last.pt

# Evaluate a trained model
python treino_yolov8seg.py --avaliar runs/segment/viaturas_eb/weights/best.pt

# Run inference on one image
python treino_yolov8seg.py --avaliar runs/segment/viaturas_eb/weights/best.pt --inferir path/to/image.jpg
```

| Option          | Default        | Description                                                |
|-----------------|----------------|------------------------------------------------------------|
| `--coco`        | —              | Folder with the COCO JSONs                                 |
| `--imagens`     | —              | Image repository root                                      |
| `--dataset`     | `dataset_yolo` | Output folder for the YOLO dataset                         |
| `--epochs`      | `0`            | Epochs (`0` = use the profile's recommendation)            |
| `--perfil`      | interactive    | Force a hardware profile (skips the menu)                  |
| `--nome-run`    | `viaturas_eb`  | Experiment name under `runs/segment/`                      |
| `--so-preparar` | off            | Convert the dataset only, do not train                     |
| `--retomar`     | —              | Resume from a checkpoint                                   |
| `--avaliar`     | —              | Evaluate a model (`best.pt`)                               |
| `--inferir`     | —              | With `--avaliar`: run inference on an image                |

### Hardware profiles

The script detects your GPU/VRAM/CPU count, recommends a profile, and lets you override it.

| Profile      | Hardware                 | Model            | Batch | Image size | Workers | Epochs (rec.) | AMP |
|--------------|--------------------------|------------------|-------|------------|---------|---------------|-----|
| `cpu_fraca`  | CPU only, < 8 cores      | `yolov8n-seg.pt` | 4     | 512        | 2       | 50            | no  |
| `cpu_forte`  | CPU only, ≥ 8 cores      | `yolov8s-seg.pt` | 8     | 640        | 8       | 250           | no  |
| `gpu_baixa`  | GPU < 6 GB VRAM          | `yolov8s-seg.pt` | 8     | 640        | 4       | 100           | yes |
| `gpu_media`  | GPU 6–12 GB VRAM         | `yolov8m-seg.pt` | 16    | 640        | 8       | 150           | yes |
| `gpu_alta`   | GPU 12+ GB VRAM          | `yolov8l-seg.pt` | 32    | 1024       | 8       | 200           | yes |

### Training details

- **Split:** 80% train / 20% validation per class, with a fixed seed (42) for reproducibility. Images without annotations are skipped.
- **Augmentation:** HSV jitter, rotation (±10°), translation, scale, shear, horizontal flip, mosaic, mixup and copy-paste. No vertical flip. On `cpu_fraca`, mosaic is halved and mixup/copy-paste are disabled for speed.
- **Early stopping:** `patience=30`; checkpoints saved every 10 epochs.
- **Metrics reported:** mAP50 and mAP50-95 for boxes and masks, plus per-class mask mAP50.

### Outputs

- `dataset_yolo/` — YOLO-seg dataset and `dataset.yaml`
- `runs/segment/<nome-run>/weights/best.pt` and `last.pt`
- Training curves and validation plots in the same run folder

### Export for deployment

```bash
yolo export model=runs/segment/viaturas_eb/weights/best.pt format=onnx
```

---

## Stage 4 — Export for Raspberry Pi (`export_rasp.py`)

Converts the trained PyTorch weights to **NCNN**, a lightweight inference format that generally gives the best CPU performance on ARM boards such as the Raspberry Pi.

```python
model = YOLO("runs/segment/runs/segment/viaturas_eb-3/weights/best.pt")
model.export(format="ncnn", imgsz=320)
```

**Usage**

1. Edit the path in `export_rasp.py` so it points to your own `best.pt` (the default is a hard-coded run folder, `viaturas_eb-3`).
2. Run it on the training machine:

   ```bash
   pip install ultralytics ncnn
   python export_rasp.py
   ```

3. Ultralytics creates a `best_ncnn_model/` folder next to `best.pt`. It contains the NCNN `.param` / `.bin` files and a `metadata.yaml` with the class names.
4. Copy that whole folder to the Raspberry Pi, next to `yolo_viat_eb.py`.

**Notes**

- `imgsz=320` trades accuracy for speed. The model is trained at 512–1024 px depending on the profile, so validate the exported model on real images and, if small or distant vehicles are missed, retry with `imgsz=416` or `480`. Use the same value in stage 5.
- Training already at the target resolution (for example, editing `imgsz` in the profile) usually recovers part of the accuracy lost by exporting at a smaller size.

---

## Stage 5 — Real-time inference on the Raspberry Pi (`yolo_viat_eb.py`)

Loads the NCNN model and runs live segmentation on webcam frames, drawing boxes, masks and class labels.

**Install (on the Pi)**

```bash
pip install ultralytics ncnn opencv-python
```

**Usage**

```bash
python yolo_viat_eb.py
```

Press `q` in the video window to quit.

**How it works**

- Loads `best_ncnn_model` with `task="segment"` (the folder must be in the working directory, or change the path in the script).
- Opens camera `0` and requests 640×480 to avoid saturating the USB bus. Try `1`, `2`, … in `cv2.VideoCapture(...)` if you have several cameras.
- Runs inference with `imgsz=320` and `stream=True` (avoids accumulating frames in RAM), then uses `r.plot()` to render the annotated frame in an OpenCV window titled "Detector de Viaturas EB - Raspberry Pi 4".

**Notes**

- `cv2.imshow` needs a display (monitor, VNC or X forwarding); it will not work over a plain headless SSH session.
- `imgsz` here must match the value used in `export_rasp.py`.
- To change sensitivity, pass `conf=0.25` (or another value) to the `model(...)` call.

---

## Typical end-to-end run

```bash
# 1. Collect and deduplicate images
python repositorio_viaturas.py --images 50

# 2. One-time setup, then auto-annotate
python pre_anotacao_gsam.py --instalar
python pre_anotacao_gsam.py --repo repositorio_viaturas --preview

# 3. Review and correct masks in CVAT / Label Studio, export back to anotacoes_coco/

# 4. Train and evaluate
python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas

# 5. Export for the Raspberry Pi (edit the .pt path in the script first)
python export_rasp.py

# 6. On the Pi: copy best_ncnn_model/ next to the script and run
python yolo_viat_eb.py
```

## Troubleshooting

| Problem                                       | Fix                                                                                   |
|-----------------------------------------------|---------------------------------------------------------------------------------------|
| ChromeDriver fails to start                   | `pip install --upgrade selenium webdriver-manager`; make sure Chrome is installed     |
| Few images returned by Bing                   | Run with `--visible` to inspect; the script falls back to Google automatically        |
| `groundingdino` not found / `BertModel` error | `python pre_anotacao_gsam.py --corrigir` (pins `transformers==4.35.2`)                |
| Git clone fails during setup                  | Install Git and make sure it is on `PATH`                                             |
| CUDA out of memory                            | Use a smaller SAM profile (`--cpu` for annotation) or a lower training profile        |
| Poor mAP on some classes                      | Add more reviewed images for that class; check for mislabeled auto-annotations        |
| `Could not open the webcam` on the Pi         | Check the camera index in `cv2.VideoCapture`, cables, and that no other app uses it   |
| `cv2.imshow` error / no window on the Pi      | Run with a display attached (or VNC); headless SSH cannot show OpenCV windows         |
| Model not found on the Pi                     | Copy the full `best_ncnn_model/` folder and run from its parent directory             |
| Low FPS on the Pi                             | Keep `imgsz=320`, lower the camera resolution, or use a smaller model (`yolov8n-seg`) |

## Notes on data quality

With roughly 30–50 images per class, expect modest results. For better accuracy: collect more varied images (angles, lighting, terrain, distance), review every auto-generated mask, and balance the number of images per class.
