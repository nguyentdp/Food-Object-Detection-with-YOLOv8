# Colab Commands

```python
!nvidia-smi
!pip install ultralytics==8.2.103
import ultralytics
ultralytics.checks()
```

```python
!yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=True
```
