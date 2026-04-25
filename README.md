# Food-Object-Detection-with-YOLOv8

Portfolio-ready INFO 4604 project using YOLOv8 object detection.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run full project
```bash
python src/run_detection.py --manifest data/image_manifest.csv --model yolov8n.pt --conf 0.25
```

## Preview page
```bash
python -m http.server 8000 --directory docs
```
Open http://localhost:8000/
