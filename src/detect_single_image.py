import argparse
from ultralytics import YOLO
p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--model",default="yolov8n.pt"); p.add_argument("--conf",type=float,default=0.25); a=p.parse_args()
model=YOLO(a.model); results=model.predict(source=a.source,conf=a.conf,save=True)
for r in results:
    boxes=r.boxes
    if boxes is None or len(boxes)==0: print("No objects detected."); continue
    for i in range(len(boxes)):
        cid=int(boxes.cls[i]); print(f"{r.names[cid]}: {float(boxes.conf[i]):.3f}")
print("Annotated image saved in runs/detect/predict*/")
