from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import cv2
import os
import shutil
import uuid
from ultralytics import YOLO

# =============================
# CONFIG
# =============================

MODEL_PATH = "best.pt"

UPLOAD_DIR = "temp"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Detection thresholds
FIRE_CONF = 0.40
SMOKE_CONF = 0.35

# Area thresholds (relative to frame area)
# دي أهم إضافة لتقليل إنذار نار صغيرة زي البوتجاز
MIN_FIRE_AREA_RATIO = 0.02
MIN_SMOKE_AREA_RATIO = 0.03

# Final decision durations
MIN_FIRE_TIME_SEC = 1.0
MIN_SMOKE_TIME_SEC = 2.0

model = YOLO(MODEL_PATH)

# =============================
# FASTAPI
# =============================

app = FastAPI(title="🔥 Fire & Smoke Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # للتجربة فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نفس المسار للفيديو والصور الناتجة
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/")
def home():
    return {"message": "API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =============================
# UTILS
# =============================

def normalize_class(name: str) -> str:
    name = name.lower().strip()

    if "fire" in name or "flame" in name:
        return "fire"
    if "smoke" in name:
        return "smoke"

    return "other"


def safe_box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1) * max(0, y2 - y1)


# =============================
# VIDEO PROCESSING
# =============================

def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = max(1, width * height)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (width, height)
    )

    fire_frames = 0
    smoke_frames = 0
    max_conf = 0.0

    largest_fire_area_ratio = 0.0
    largest_smoke_area_ratio = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=0.25)[0]
        boxes = results.boxes

        fire_detected = False
        smoke_detected = False

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            raw_class = model.names[cls_id]
            class_name = normalize_class(raw_class)

            box_area = safe_box_area(x1, y1, x2, y2)
            area_ratio = box_area / frame_area

            if class_name == "fire":
                largest_fire_area_ratio = max(largest_fire_area_ratio, area_ratio)

            if class_name == "smoke":
                largest_smoke_area_ratio = max(largest_smoke_area_ratio, area_ratio)

            # النار لا تُحسب خطر إلا لو حجمها كبير كفاية
            if class_name == "fire" and conf >= FIRE_CONF and area_ratio >= MIN_FIRE_AREA_RATIO:
                fire_detected = True
                max_conf = max(max_conf, conf)

            # الدخان برضو بحجم مناسب
            if class_name == "smoke" and conf >= SMOKE_CONF and area_ratio >= MIN_SMOKE_AREA_RATIO:
                smoke_detected = True
                max_conf = max(max_conf, conf)

            # الرسم على كل detections لعرض أفضل، حتى لو صغيرة
            color = (0, 0, 255) if class_name == "fire" else (0, 255, 0)
            label = f"{class_name} {conf:.2f} | area:{area_ratio:.3f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        if fire_detected:
            fire_frames += 1
        if smoke_detected:
            smoke_frames += 1

        writer.write(frame)

    cap.release()
    writer.release()

    fire_time = fire_frames / fps
    smoke_time = smoke_frames / fps

    fire_exists = fire_time >= MIN_FIRE_TIME_SEC
    smoke_exists = smoke_time >= MIN_SMOKE_TIME_SEC

    alarm = False
    alarm_type = "none"

    # منطق القرار النهائي
    if fire_exists or smoke_exists:
        alarm = True

        if fire_exists and smoke_exists:
            alarm_type = "fire_and_smoke"
        elif fire_exists:
            alarm_type = "fire"
        else:
            alarm_type = "smoke"

    return {
        "alarm": alarm,
        "alarm_type": alarm_type,
        "fire_detected": fire_exists,
        "smoke_detected": smoke_exists,
        "max_confidence": round(max_conf, 3),
        "fire_duration_sec": round(fire_time, 2),
        "smoke_duration_sec": round(smoke_time, 2),
        "detection_duration_sec": round(max(fire_time, smoke_time), 2),
        "largest_fire_area_ratio": round(largest_fire_area_ratio, 4),
        "largest_smoke_area_ratio": round(largest_smoke_area_ratio, 4),
    }


# =============================
# IMAGE PROCESSING
# =============================

def process_image(input_path: str, output_path: str):
    frame = cv2.imread(input_path)
    if frame is None:
        raise Exception("Cannot read image")

    height, width = frame.shape[:2]
    frame_area = max(1, width * height)

    results = model(frame, verbose=False, conf=0.25)[0]
    boxes = results.boxes

    max_conf = 0.0
    fire_detected = False
    smoke_detected = False

    largest_fire_area_ratio = 0.0
    largest_smoke_area_ratio = 0.0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        raw_class = model.names[cls_id]
        class_name = normalize_class(raw_class)

        box_area = safe_box_area(x1, y1, x2, y2)
        area_ratio = box_area / frame_area

        if class_name == "fire":
            largest_fire_area_ratio = max(largest_fire_area_ratio, area_ratio)

        if class_name == "smoke":
            largest_smoke_area_ratio = max(largest_smoke_area_ratio, area_ratio)

        if class_name == "fire" and conf >= FIRE_CONF and area_ratio >= MIN_FIRE_AREA_RATIO:
            fire_detected = True
            max_conf = max(max_conf, conf)

        if class_name == "smoke" and conf >= SMOKE_CONF and area_ratio >= MIN_SMOKE_AREA_RATIO:
            smoke_detected = True
            max_conf = max(max_conf, conf)

        color = (0, 0, 255) if class_name == "fire" else (0, 255, 0)
        label = f"{class_name} {conf:.2f} | area:{area_ratio:.3f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imwrite(output_path, frame)

    alarm = False
    alarm_type = "none"

    if fire_detected or smoke_detected:
        alarm = True
        if fire_detected and smoke_detected:
            alarm_type = "fire_and_smoke"
        elif fire_detected:
            alarm_type = "fire"
        else:
            alarm_type = "smoke"

    return {
        "alarm": alarm,
        "alarm_type": alarm_type,
        "max_confidence": round(max_conf, 3),
        "largest_fire_area_ratio": round(largest_fire_area_ratio, 4),
        "largest_smoke_area_ratio": round(largest_smoke_area_ratio, 4),
    }


# =============================
# VIDEO ENDPOINT
# =============================

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid video format")

    file_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_out.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_video(input_path, output_path)

    return JSONResponse({
        "alarm": result["alarm"],
        "alarm_type": result["alarm_type"],
        "fire_detected": result["fire_detected"],
        "smoke_detected": result["smoke_detected"],
        "max_confidence": result["max_confidence"],
        "fire_duration_sec": result["fire_duration_sec"],
        "smoke_duration_sec": result["smoke_duration_sec"],
        "detection_duration_sec": result["detection_duration_sec"],
        "largest_fire_area_ratio": result["largest_fire_area_ratio"],
        "largest_smoke_area_ratio": result["largest_smoke_area_ratio"],
        "video_url": f"/outputs/{file_id}_out.mp4"
    })


# =============================
# IMAGE ENDPOINT
# =============================

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid image format")

    file_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}_out.jpg")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_image(input_path, output_path)

    return JSONResponse({
        "alarm": result["alarm"],
        "alarm_type": result["alarm_type"],
        "max_confidence": result["max_confidence"],
        "largest_fire_area_ratio": result["largest_fire_area_ratio"],
        "largest_smoke_area_ratio": result["largest_smoke_area_ratio"],
        "image_url": f"/outputs/{file_id}_out.jpg"
    })