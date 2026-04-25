from ultralytics import YOLOWorld

with open("src/food_classes.txt", "r") as f:
    classes = [c.strip() for c in f.read().split(",") if c.strip()]

model = YOLOWorld("yolov8m-worldv2.pt")
model.set_classes(classes)

model.predict(
    source=0,
    conf=0.55,
    show=True,
    save=False
)