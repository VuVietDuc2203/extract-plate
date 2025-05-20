from ultralytics import YOLO
import cv2
import os
import re
import queue
import threading
from datetime import datetime

# Configuration
DETECTION_MODEL_PATH = "lpd.onnx"
OCR_MODEL_PATH = "ocr.onnx"
INPUT_VIDEO = "data/2025-05-12_12-48-37_TVP_dem.mp4"
OUTPUT_DIR = "plates_results/"
OCR_LABELS = "ocr.names"
TARGET_FPS = 1
NUM_WORKER_THREADS = 6

# Global queue for frame processing
frame_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
stop_event = threading.Event()

def load_ocr_mapping():
    with open(OCR_LABELS, "r") as f:
        return f.read().splitlines()

ocr_mapping = load_ocr_mapping()

def preprocess_plate(cropped_plate):
    h, w = cropped_plate.shape[:2]
    target_size = (224,224)
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(cropped_plate, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        (target_size[0] - new_h) // 2,
        (target_size[0] - new_h) - (target_size[0] - new_h) // 2,
        (target_size[1] - new_w) // 2,
        (target_size[1] - new_w) - (target_size[1] - new_w) // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    return padded

def detect_plates(model, frame):
    results = model(frame, conf=0.7, iou=0.7)
    return [box.xyxy[0].cpu().numpy() for result in results for box in result.boxes]

def recognize_plate(model, plate_img):
    results = model(plate_img, conf=0.3)
    chars = []
    for result in results:
        for box in result.boxes:
            x, y, w, h = box.xywh[0].cpu().numpy()
            chars.append((y, x, int(box.cls)))

    if not chars:
        return ""

    y_coords = [y for y, _, _ in chars]
    sorted_chars = sorted(chars, key=lambda c: (c[0], c[1]))

    if len(y_coords) > 1:
        y_mean = sum(y_coords) / len(y_coords)
        line1 = [c for c in sorted_chars if c[0] < y_mean]
        line2 = [c for c in sorted_chars if c[0] >= y_mean]

        line1_sorted = sorted(line1, key=lambda c: c[1])
        line2_sorted = sorted(line2, key=lambda c: c[1])

        text_line1 = "".join([ocr_mapping[c[2]] for c in line1_sorted])
        text_line2 = "".join([ocr_mapping[c[2]] for c in line2_sorted])
        return f"{text_line1}_{text_line2}"
    else:
        return "".join([ocr_mapping[c[2]] for c in sorted_chars])

def process_frame(frame, det_model, ocr_model):
    for box in detect_plates(det_model, frame):
        x1, y1, x2, y2 = map(int, box)
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:  # Skip if no plate detected
            continue
        processed = preprocess_plate(plate_img)
        plate_text = recognize_plate(ocr_model, processed)

        if plate_text:
            clean_text = re.sub(r'[^A-Za-z0-9]', '_', plate_text.replace('__', '_'))
            cv2.imwrite(f"{OUTPUT_DIR}/{clean_text}.jpg", plate_img)

def worker_thread(det_model, ocr_model):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1)  # Timeout to periodically check stop_event
            process_frame(frame, det_model, ocr_model)
            frame_queue.task_done()
        except queue.Empty:
            continue

def video_reader_thread(video_path, target_fps):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        stop_event.set()
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30

    skip_frames = max(1, int(original_fps / target_fps))
    frame_count = 0

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            # Put frame in queue (block if queue is full)
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                print("Warning: Frame queue full, dropping frame")

        frame_count += 1

        if skip_frames > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    stop_event.set()  # Signal all threads to stop

if __name__ == "__main__":
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize models (once in main thread)
    det_model = YOLO(DETECTION_MODEL_PATH)
    ocr_model = YOLO(OCR_MODEL_PATH)

    # Start worker threads
    workers = []
    for _ in range(NUM_WORKER_THREADS):
        t = threading.Thread(target=worker_thread, args=(det_model, ocr_model))
        t.daemon = True
        t.start()
        workers.append(t)

    # Start video reader thread
    reader_thread = threading.Thread(target=video_reader_thread, args=(INPUT_VIDEO, TARGET_FPS))
    reader_thread.start()

    try:
        # Wait for all threads to complete
        reader_thread.join()
        frame_queue.join()  # Wait for all frames to be processed
    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
        stop_event.set()

    # Ensure all threads are stopped
    for t in workers:
        t.join(timeout=1)
    reader_thread.join(timeout=1)

    print("Processing complete")