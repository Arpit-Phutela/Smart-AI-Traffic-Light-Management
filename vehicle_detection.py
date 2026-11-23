from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.predict(
    source="traffic_video.mp4",
    save=True,
    show=True,
    conf=0.4
)