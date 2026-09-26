# Computer Vision & OCR Models Directory

This directory stores pre-trained weights for vehicle detection and license plate detection.

---

## 1. Vehicle Detection Model
- **Default Filename**: `yolov8n.pt` (or `yolov8s.pt`, `yolov8m.pt`)
- **Config Key**: `VEHICLE_MODEL_PATH` in `.env`
- **Loading Behavior**: 
  - If a local file exists at the specified path (or in the root/models folder), Ultralytics loads it directly.
  - If not found locally, Ultralytics automatically downloads the standard COCO-pretrained `yolov8n.pt` on the first run.
- **Classes Used**: `car`, `motorcycle`, `bus`, `truck`.

---

## 2. Dedicated License Plate Detection Model
- **Config Key**: `PLATE_MODEL_PATH` in `.env` (default: `models/license_plate_detector.pt`)
- **Recommended Model Type**:
  - A YOLOv8 license plate detector model trained on license plate datasets (e.g. `license_plate_detector.pt`, `best.pt`).
- **Loading Behavior**:
  - The pipeline checks if the file specified by `PLATE_MODEL_PATH` exists on disk.
  - **If found**: Loads the YOLO model to detect exact license plate bounding boxes on vehicles with high precision.
  - **If not found**: The pipeline logs:
    `[INFO] [CV]: PLATE DETECTOR: No custom model found at 'models/license_plate_detector.pt'. Using selective vehicle ROI & aspect-ratio plate candidate extractor as fallback.`
    and activates the selective vehicle ROI candidate extractor so that testing and execution remain 100% operational.

---

## 3. Optical Character Recognition (OCR) Engine
- **Primary Engine**: **PaddleOCR** (`PP-OCRv6` / `PP-LCNet` lightweight models).
  - Automatically configured with angle classification and multi-pass enhancement.
- **Fallback Engine**: **EasyOCR** (`easyocr.Reader(['en'])`).
  - Seamlessly activates if PaddleOCR is unavailable in the environment.
- **Status Logging**: Clearly logs the active engine on startup:
  `[INFO] [OCR]: OCR ENGINE: PaddleOCR initialized successfully.`

---

## 4. Diagnostic & Debug Settings (in `.env`)
```ini
# Enable/disable terminal diagnostic output for plate detection & OCR
PLATE_DEBUG=True

# Enable/disable temporary debug image saving (debug/plates, debug/enhanced, debug/failed)
DEBUG_PLATE_IMAGES=True

# Tunable thresholds
PLATE_CONFIDENCE_THRESHOLD=0.35
OCR_CONFIDENCE_THRESHOLD=0.55
MAX_OCR_ATTEMPTS=10
OCR_FRAME_INTERVAL=3
```

---

## How to add custom weights:
1. Copy your trained plate detection `.pt` file into this `models/` directory (e.g. `models/license_plate_detector.pt`).
2. Verify the path in your `.env` file:
   ```ini
   PLATE_MODEL_PATH=models/license_plate_detector.pt
   ```
3. Restart the application.
