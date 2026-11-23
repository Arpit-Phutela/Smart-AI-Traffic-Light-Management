from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")


vehicle_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture("traffic_video.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

line_y = height // 2

counted_ids = set()

vehicle_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        conf=0.4
    )


    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 255, 255),
        3
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Center of vehicle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Tracking ID
            cv2.putText(
                frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Count when vehicle crosses line
            if center_y > line_y and track_id not in counted_ids:

                counted_ids.add(track_id)
                vehicle_count += 1

    # Display count
    cv2.putText(
        frame,
        f"Vehicles Counted: {vehicle_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3
    )

    cv2.imshow(
        "Vehicle Counting",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()