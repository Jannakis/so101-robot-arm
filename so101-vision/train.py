"""Trainiert YOLO26n auf dem Schreibtisch-/Brett-Datensatz.

Aufruf (aus so101-vision/):
    .venv\\Scripts\\python.exe train.py
    .venv\\Scripts\\python.exe train.py --epochs 50 --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained Checkpoint")
    parser.add_argument("--data", default=str(ROOT / "configs" / "objects.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu' oder GPU-Index, z.B. '0'")
    parser.add_argument("--name", default="yolo26n_recipe")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=100,  # Early Stopping, wenn sich die Fitness 100 Epochen nicht verbessert
        cache="ram",  # 53 Bilder passen locker in den RAM
        workers=2,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
        # Augmentationen wie im Roboflow-Trainingsrezept
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,  # keine Rotation: sonst werden die Boxen dünner Objekte (Marker) zu groß
        translate=0.1,
        scale=0.5,
        mosaic=1.0,
        hsv_h=0.028,  # Hue ±5°
        hsv_s=0.5,  # Saturation ±50 %
        hsv_v=0.2,  # Brightness ±20 %
        plots=True,
    )

    # Finale Auswertung des besten Checkpoints auf dem Test-Split
    best = YOLO(ROOT / "runs" / args.name / "weights" / "best.pt")
    metrics = best.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device,
                       project=str(ROOT / "runs"), name=f"{args.name}_test", exist_ok=True)
    print(f"\nTest mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
