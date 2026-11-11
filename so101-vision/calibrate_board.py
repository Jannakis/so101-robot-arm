"""Kalibriert die Umrechnung Bild-Pixel -> Brett-Koordinaten über das Schachbrett.

Brett-Koordinatensystem (Blick von oben, Roboter steht unten im Bild):
    Ursprung = untere linke Ecke des 8x8-Spielfelds (Roboterseite)
    X        = nach rechts
    Y        = vom Roboter weg (im Bild nach oben)
    Einheit  = Schachfelder; mit --square-mm zusätzlich Millimeter

Die Kamera muss danach fest stehen bleiben. Wird sie bewegt: neu kalibrieren.

Aufruf (aus so101-vision/):
    .venv\\Scripts\\python.exe calibrate_board.py data/raw/img_007.jpg
    .venv\\Scripts\\python.exe calibrate_board.py data/raw/img_007.jpg --square-mm 45
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CALIB = ROOT / "configs" / "board_calibration.json"
INNER = 7  # 8x8 Felder -> 7x7 innere Ecken


def find_inner_corners(image: np.ndarray) -> np.ndarray:
    """Findet die 7x7 inneren Ecken, sortiert: Zeile 0 = oben im Bild, Spalte 0 = links."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (INNER, INNER), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not found:
        raise RuntimeError("Schachbrett nicht gefunden: Brett möglichst frei und vollständig im Bild?")
    grid = corners.reshape(INNER, INNER, 2)
    # OpenCV liefert die Ecken je nach Bild in unterschiedlicher Reihenfolge -> vereinheitlichen
    step = grid[0, 1] - grid[0, 0]
    if abs(step[0]) < abs(step[1]):
        grid = grid.transpose(1, 0, 2)  # Spalten sollen entlang der Bild-x-Achse laufen
    if grid[0, -1, 0] < grid[0, 0, 0]:
        grid = grid[:, ::-1]  # links -> rechts
    if grid[-1, 0, 1] < grid[0, 0, 1]:
        grid = grid[::-1]  # oben -> unten
    return grid


def board_points() -> np.ndarray:
    """Brett-Koordinaten (in Feldern) der inneren Ecken, gleiche Sortierung wie find_inner_corners."""
    rows, cols = np.mgrid[0:INNER, 0:INNER]
    return np.stack([cols + 1, INNER - rows], axis=-1).astype(np.float64)


class BoardCalibration:
    """Lädt die gespeicherte Kalibrierung und rechnet Pixel in Brett-Koordinaten um."""

    def __init__(self, path: Path = DEFAULT_CALIB):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.H = np.array(data["homography"], dtype=np.float64)
        self.image_size = tuple(data["image_size"])  # (Breite, Höhe)
        self.square_mm = data.get("square_mm")

    def to_board(self, x: float, y: float) -> tuple[float, float]:
        """Pixel (x, y) -> Brett-Koordinaten in Feldern."""
        p = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), self.H)[0, 0]
        return float(p[0]), float(p[1])

    def format(self, x: float, y: float) -> str:
        bx, by = self.to_board(x, y)
        if self.square_mm:
            return f"board=({bx * self.square_mm:.0f}, {by * self.square_mm:.0f}) mm"
        return f"board=({bx:.2f}, {by:.2f}) Felder"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Foto mit gut sichtbarem Schachbrett, aus der festen Kameraposition")
    parser.add_argument("--square-mm", type=float, default=None, help="Kantenlänge eines Feldes in mm")
    parser.add_argument("--out", default=str(DEFAULT_CALIB))
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    img_pts = find_inner_corners(image).reshape(-1, 2)
    brd_pts = board_points().reshape(-1, 2)

    H, _ = cv2.findHomography(img_pts, brd_pts)
    reproj = cv2.perspectiveTransform(brd_pts.reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)
    err_px = float(np.linalg.norm(reproj - img_pts, axis=1).mean())

    h, w = image.shape[:2]
    Path(args.out).write_text(json.dumps({
        "source_image": str(Path(args.image).name),
        "image_size": [w, h],
        "square_mm": args.square_mm,
        "reprojection_error_px": round(err_px, 3),
        "homography": H.tolist(),
    }, indent=2), encoding="utf-8")

    # Kontrollbild: das ideale 8x8-Raster zurück ins Foto projiziert, Ursprung und Achsen markiert
    H_inv = np.linalg.inv(H)
    to_px = lambda pts: cv2.perspectiveTransform(np.array(pts, dtype=np.float64).reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    vis = image.copy()
    for i in range(9):
        for a, b in (((i, 0), (i, 8)), ((0, i), (8, i))):
            p, q = to_px([a, b]).astype(int)
            cv2.line(vis, tuple(p), tuple(q), (0, 255, 0), 2)
    o, ex, ey = to_px([(0, 0), (1.5, 0), (0, 1.5)]).astype(int)
    cv2.arrowedLine(vis, tuple(o), tuple(ex), (0, 0, 255), 5, tipLength=0.2)
    cv2.arrowedLine(vis, tuple(o), tuple(ey), (255, 0, 0), 5, tipLength=0.2)
    cv2.putText(vis, "X", tuple(ex + [10, 10]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(vis, "Y", tuple(ey + [10, 0]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
    vis_path = ROOT / "runs" / "board_calibration.jpg"
    vis_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(vis_path), vis)

    print(f"Kalibrierung gespeichert: {args.out}")
    print(f"Mittlerer Reprojektionsfehler: {err_px:.2f} px")
    print(f"Kontrollbild: {vis_path}")
    if not args.square_mm:
        print("Hinweis: ohne --square-mm werden Koordinaten in Schachfeldern ausgegeben.")


if __name__ == "__main__":
    main()
