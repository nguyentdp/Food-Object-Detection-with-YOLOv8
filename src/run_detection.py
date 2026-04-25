import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLOWorld


def load_food_classes(path):
    with open(path, "r") as f:
        return [c.strip() for c in f.read().split(",") if c.strip()]


def rows(result, image_id, source):
    out = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for i in range(len(boxes)):
        cid = int(cls[i])
        x1, y1, x2, y2 = xyxy[i]

        out.append({
            "image_id": image_id,
            "source": source,
            "object": result.names[cid],
            "class_id": cid,
            "confidence": float(conf[i]),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "box_area": float((x2 - x1) * (y2 - y1))
        })

    return out


p = argparse.ArgumentParser()
p.add_argument("--manifest", default="data/image_manifest.csv")
p.add_argument("--model", default="yolov8m-worldv2.pt")
p.add_argument("--classes", default="food_classes.txt")
p.add_argument("--conf", type=float, default=0.55)
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()

for d in ["reports", "reports/figures", "docs/assets"]:
    Path(d).mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(a.manifest)

food_classes = load_food_classes(a.classes)

model = YOLOWorld(a.model)
model.set_classes(food_classes)

all_rows = []

for _, row in manifest.iterrows():
    print("Detecting", row["image_id"])

    results = model.predict(
        source=str(row["source"]),
        conf=a.conf,
        imgsz=a.imgsz,
        save=True,
        stream=False
    )

    for r in results:
        all_rows += rows(r, str(row["image_id"]), str(row["source"]))

df = pd.DataFrame(all_rows)
df.to_csv("reports/detections.csv", index=False)

summary = {
    "images_processed": len(manifest),
    "total_detections": len(df),
    "average_confidence": None if df.empty else round(float(df.confidence.mean()), 4),
    "unique_objects": 0 if df.empty else int(df.object.nunique()),
    "top_objects": {} if df.empty else df.object.value_counts().head(10).to_dict(),
    "confidence_threshold": a.conf,
    "model": a.model,
    "classes_file": a.classes
}

Path("reports/summary.json").write_text(json.dumps(summary, indent=2))

if not df.empty:
    df.object.value_counts().head(12).sort_values().plot(
        kind="barh",
        figsize=(9, 5),
        title="Most Frequently Detected Food Objects"
    )
    plt.tight_layout()
    plt.savefig("reports/figures/class_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.hist(df.confidence, bins=15)
    plt.title("Detection Confidence Scores")
    plt.tight_layout()
    plt.savefig("reports/figures/confidence_histogram.png", dpi=180)
    plt.close()

for f in Path("reports/figures").glob("*.png"):
    shutil.copy(f, Path("docs/assets") / f.name)

print("Done. See reports/detections.csv and docs/index.html")