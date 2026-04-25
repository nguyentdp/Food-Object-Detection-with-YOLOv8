# Food Object Detection with YOLO-World

This project uses YOLO-World to detect food objects from images and a live webcam. It focuses on simple, clear food items like apple, banana, donut, pizza, and orange.

## Features

- Detect food objects from images
- Live webcam detection
- Custom food labels using `food_classes.txt`
- Outputs results to CSV and charts

## Run

Install:

```bash 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run detection on images:

```bash 
python src/run_detection.py --manifest data/image_manifest.csv
```

Run webcam:

```bash 
python src/camera_food_detection.py
```

View page:

```bash 
python -m http.server 8000 --directory docs
```

Open:
http://localhost:8000/
