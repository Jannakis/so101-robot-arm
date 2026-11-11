"""Erkennt Objekte mit dem trainierten Modell und gibt Mittelpunkte in Pixeln
und (falls kalibriert, siehe calibrate_board.py) in Brett-Koordinaten aus.

Aufruf (aus so101-vision/):
    .venv\\Scripts\\python.exe predict.py data/raw/img_001.jpg
    .venv\\Scripts\\python.exe predict.py 0          # Webcam-Index 0, live
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from calibrate_board import DEFAULT_CALIB, BoardCalibration

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "yolo26n_4cls_fixed" / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Bild, Ordner, Video oder Kamera-Index")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--targets", nargs="+", default=["marker", "tape"],
                        help="Nur diese Klassen ausgeben (Rest sind Störobjekte)")
    parser.add_argument("--all", action="store_true", help="Alle Klassen ausgeben")
    parser.add_argument("--calib", default=str(DEFAULT_CALIB), help="Brett-Kalibrierung (JSON)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    name_to_id = {name: i for i, name in model.names.items()}
    classes = None if args.all else [name_to_id[t] for t in args.targets]

    board = BoardCalibration(args.calib) if Path(args.calib).exists() else None
    if board is None:
        print("Keine Brett-Kalibrierung gefunden -> nur Pixel-Koordinaten (siehe calibrate_board.py)")

    for result in model.predict(source, conf=args.conf, imgsz=args.imgsz, classes=classes, stream=True,
                                show=isinstance(source, int), save=not isinstance(source, int),
                                project=str(ROOT / "runs"), name="predict", exist_ok=True):
        print(Path(result.path).name)
        h, w = result.orig_shape
        if board and (w, h) != board.image_size:
            print(f"  WARNUNG: Bild {w}x{h}, kalibriert wurde mit {board.image_size[0]}x{board.image_size[1]}"
                  " -> Brett-Koordinaten stimmen nicht")
        for box in result.boxes:
            cx, cy, bw, bh = box.xywh[0].tolist()
            name = model.names[int(box.cls)]
            line = f"  {name:<8} conf={float(box.conf):.2f}  center=({cx:.0f}, {cy:.0f}) px  size={bw:.0f}x{bh:.0f}"
            if board:
                line += f"  {board.format(cx, cy)}"
            print(line)


if __name__ == "__main__":
    main()
