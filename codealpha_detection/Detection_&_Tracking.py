import cv2
import time
import numpy as np

from ultralytics import YOLOWorld
from deep_sort_realtime.deepsort_tracker import DeepSort

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

SOURCE = 0

MODEL_PATH = "yolov8s-worldv2.pt"
IMG_SIZE = 416
CONFIDENCE = 0.35
IOU = 0.45

CLASSES = [
    "person",
    "cell phone",
    "pen",
    "pencil",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "bottle",
    "cup",
    "chair",
    "table",
    "backpack",
    "watch",
    "headphones",
    "remote control",
    "wallet",
    "keys"
]

print("Loading YOLO-World...")
model = YOLOWorld(MODEL_PATH)
model.set_classes(CLASSES)
print("YOLO-World loaded successfully.")

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(
            dim_x=7,
            dim_z=4
        )

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=float)

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=float)

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def convert_bbox_to_z(self, bbox):
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.0
        y = y1 + h / 2.0
        s = w * h
        r = w / float(h + 1e-6)

        return np.array([
            [x],
            [y],
            [s],
            [r]
        ])

    def convert_x_to_bbox(self, x):
        x = np.asarray(x).reshape(-1)
    
        x_pos = float(x[0])
        y_pos = float(x[1])
        s = float(x[2])
        r = float(x[3])
        s = max(s, 1e-6)
        r = max(r, 1e-6)

        w = np.sqrt(abs(s * r))
        h = s / (w + 1e-6)
        return np.array([
            x_pos - w / 2.0,
            y_pos - h / 2.0,
            x_pos + w / 2.0,
            y_pos + h / 2.0
        ])

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(
            self.convert_bbox_to_z(bbox)
        )

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1

        return self.convert_x_to_bbox(
            self.kf.x
        )

    def get_state(self):
        return self.convert_x_to_bbox(
            self.kf.x
        )

def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty(
            (
                len(bb_test),
                len(bb_gt)
            )
        )

    result = np.zeros(
        (
            len(bb_test),
            len(bb_gt)
        )
    )

    for i, a in enumerate(bb_test):
        ax1, ay1, ax2, ay2 = a
        area_a = (
            max(0, ax2 - ax1)
            *
            max(0, ay2 - ay1)
        )

        for j, b in enumerate(bb_gt):
            bx1, by1, bx2, by2 = b
            area_b = (
                max(0, bx2 - bx1)
                *
                max(0, by2 - by1)
            )

            xx1 = max(ax1, bx1)
            yy1 = max(ay1, by1)
            xx2 = min(ax2, bx2)
            yy2 = min(ay2, by2)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            intersection = w * h
            union = area_a + area_b - intersection
            if union > 0:

                result[i, j] = (
                    intersection / union
                )

    return result

class SortTracker:
    def __init__(
        self,
        max_age=20,
        min_hits=2,
        iou_threshold=0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        predicted = []
        for tracker in self.trackers:
            prediction = tracker.predict()
            predicted.append(prediction)
        if len(predicted) > 0:
            predicted = np.array(predicted)

        else:
            predicted = np.empty(
                (0, 4)
            )

        if len(detections) == 0:
            matched = []
            unmatched_detections = []
            unmatched_trackers = list(
                range(len(predicted))
            )

        elif len(predicted) == 0:
            matched = []
            unmatched_detections = list(
                range(len(detections))
            )

            unmatched_trackers = []
        else:

            iou_matrix = iou_batch(
                detections,
                predicted
            )

            rows, cols = linear_sum_assignment(
                -iou_matrix
            )

            matched = []
            unmatched_detections = []
            unmatched_trackers = []
            for d in range(len(detections)):
                if d not in rows:
                    unmatched_detections.append(d)

            for t in range(len(predicted)):
                if t not in cols:
                    unmatched_trackers.append(t)

            for r, c in zip(rows, cols):
                if (
                    iou_matrix[r, c]
                    <
                    self.iou_threshold
                ):
                    unmatched_detections.append(r)
                    unmatched_trackers.append(c)

                else:
                    matched.append((r, c))

        for detection_index, tracker_index in matched:
            self.trackers[
                tracker_index
            ].update(
                detections[detection_index]
            )

        for detection_index in unmatched_detections:
            tracker = KalmanBoxTracker(
                detections[detection_index]
            )
            self.trackers.append(tracker)

        output = []
        for tracker in self.trackers:
            if (
                tracker.time_since_update < 1
                and
                (
                    tracker.hit_streak >= self.min_hits
                    or
                    self.frame_count <= self.min_hits
                )
            ):

                bbox = tracker.get_state()
                output.append(
                    (
                        bbox,
                        tracker.id
                    )
                )

        self.trackers = [
            tracker
            for tracker in self.trackers
            if tracker.time_since_update <= self.max_age
        ]
        return output

sort_tracker = SortTracker(
    max_age=20,
    min_hits=2,
    iou_threshold=0.3
)

deep_sort = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.4,
    nn_budget=100
)

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Camera open nahi hua")
    raise RuntimeError(
        "Camera open nahi ho raha"
    )

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)
cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)
previous_time = time.time()
fps = 0.0

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Not able to capture frame")
            continue

        if SOURCE == 0:

            frame = cv2.flip(
                frame,
                1
            )

        results = model(
            frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            iou=IOU,
            max_det=30,
            verbose=False
        )

        result = results[0]
        sort_detections = []
        deep_detections = []

        if result.boxes is not None:
            for box in result.boxes:
                confidence = float(
                    box.conf[0]
                )
                if confidence < CONFIDENCE:
                    continue

                class_id = int(
                    box.cls[0]
                )

                object_name = model.names[
                    class_id
                ]
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )
                sort_detections.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                )
                deep_detections.append(
                    (
                        [
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1
                        ],
                        confidence,
                        object_name
                    )
                )

        sort_results = sort_tracker.update(
            sort_detections
        )

        deep_results = deep_sort.update_tracks(
            deep_detections,
            frame=frame
        )

        deep_boxes = []
        for track in deep_results:
            if not track.is_confirmed():
                continue
            if track.time_since_update > 1:
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(
                int,
                ltrb
            )
            deep_boxes.append(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                    track_id
                )
            )

        object_count = 0

        for (
            x1,
            y1,
            x2,
            y2,
            deep_id
        ) in deep_boxes:
            best_name = "object"
            best_conf = 0.0
            best_iou = 0.0

            for detection in deep_detections:
                bbox = detection[0]
                confidence = detection[1]
                object_name = detection[2]
                dx, dy, dw, dh = bbox
                dx2 = dx + dw
                dy2 = dy + dh
                ix1 = max(
                    x1,
                    dx
                )

                iy1 = max(
                    y1,
                    dy
                )

                ix2 = min(
                    x2,
                    dx2
                )

                iy2 = min(
                    y2,
                    dy2
                )

                iw = max(
                    0,
                    ix2 - ix1
                )

                ih = max(
                    0,
                    iy2 - iy1
                )

                intersection = iw * ih
                area1 = (
                    max(0, x2 - x1)
                    *
                    max(0, y2 - y1)
                )

                area2 = (
                    max(0, dx2 - dx)
                    *
                    max(0, dy2 - dy)
                )

                union = (
                    area1
                    +
                    area2
                    -
                    intersection
                )

                current_iou = (
                    intersection / union
                    if union > 0
                    else 0
                )


                if current_iou > best_iou:
                    best_iou = current_iou
                    best_name = object_name
                    best_conf = confidence

            best_sort_id = "-"
            best_sort_iou = 0.0

            for sort_bbox, sort_id in sort_results:

                sx1, sy1, sx2, sy2 = map(
                    int,
                    sort_bbox
                )

                ix1 = max(
                    x1,
                    sx1
                )

                iy1 = max(
                    y1,
                    sy1
                )

                ix2 = min(
                    x2,
                    sx2
                )

                iy2 = min(
                    y2,
                    sy2
                )

                iw = max(
                    0,
                    ix2 - ix1
                )

                ih = max(
                    0,
                    iy2 - iy1
                )

                intersection = iw * ih

                area1 = (
                    max(0, x2 - x1)
                    *
                    max(0, y2 - y1)
                )

                area2 = (
                    max(0, sx2 - sx1)
                    *
                    max(0, sy2 - sy1)
                )

                union = (
                    area1
                    +
                    area2
                    -
                    intersection
                )

                current_iou = (
                    intersection / union
                    if union > 0
                    else 0
                )

                if current_iou > best_sort_iou:

                    best_sort_iou = current_iou

                    best_sort_id = sort_id


            object_count += 1
            label = (
                f"{best_name} | "
                f"{best_conf * 100:.0f}% | "
                f"SORT:{best_sort_id} | "
                f"DeepSORT:{deep_id}"
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                2
            )


            label_y = max(
                y1,
                text_height + 10
            )

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y - text_height - 10
                ),
                (
                    x1 + text_width,
                    label_y + baseline - 5
                ),
                (0, 255, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    label_y - 5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 0),
                2
            )
        current_time = time.time()

        difference = (
            current_time
            -
            previous_time
        )

        if difference > 0:
            fps = 1.0 / difference
        previous_time = current_time

        cv2.rectangle(
            frame,
            (0, 0),
            (640, 70),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Objects: {object_count}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Task 4 - YOLO + SORT + DeepSORT",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

except KeyboardInterrupt:
    print(
        "\nProgram stopped by user."
    )

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

print(
    "Camera closed successfully."
)