# Vehicle Detection & Parking Allocation Component

A modular, continuous computer vision and database verification component built with Python, Flask, OpenCV, Ultralytics YOLO, EasyOCR, and MySQL.

---

## 1. System Architecture

```
VIDEO STREAM (Camera or Uploaded File)
                ↓
    Vehicle Detection (YOLOv8)
                ↓
    Vehicle Tracking (ByteTrack)
                ↓
    Unique Vehicle Counter
                ↓
   License Plate Detection & ROI
                ↓
         EasyOCR Engine
                ↓
      Plate Normalization
                ↓
  MySQL Verification (previous_bookings)
                ↓
     Parking Allocation (YES / NO)
                ↓
  Terminal Output + Dashboard Display
```

---

## 2. Directory Structure

```
vehicle_detection/
├── app.py                  # Flask server, streaming & REST API endpoints
├── detection.py            # YOLO detection, tracking, plate OCR & worker thread
├── database.py             # MySQL connector, booking checks, and history logging
├── database.sql            # Schema initialization and sample booking records
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables configuration template
├── .gitignore              # Git ignore rules
├── models/
│   └── README.md           # Model setup and weights guide
├── uploads/
│   └── .gitkeep            # Video uploads directory
├── templates/
│   └── dashboard.html      # HTML dashboard UI
└── static/
    ├── css/
    │   └── style.css       # Clean dark-themed CSS styling
    └── js/
        └── dashboard.js    # Telemetry polling and control scripts
```

---

## 3. First-Run Setup Guide (Windows)

### Step 1: Install Python 3.10+
Ensure Python 3.10 or newer is installed with `pip` and added to your system PATH.

### Step 2: Open Terminal & Navigate to Project Directory
```powershell
cd "e:\Vehichle detection"
```

### Step 3: Create & Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Setup MySQL Database
1. Ensure MySQL Server is running on port 3306.
2. Execute the `database.sql` initialization script via MySQL CLI or MySQL Workbench:
```powershell
mysql -u root -p < database.sql
```
*(Enter your MySQL root password when prompted)*

### Step 6: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` and edit your MySQL credentials:
```ini
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=parking_system
```

### Step 7: Model Preparation
- By default, the system will automatically download `yolov8n.pt` on first startup if it is not already present.
- If you have a custom trained YOLO license plate detection model (e.g. `license_plate_detector.pt`), place it in the `models/` directory.
- (See [models/README.md](file:///e:/Vehichle%20detection/models/README.md) for details).

### Step 8: Start the Flask Application
```powershell
python app.py
```

### Step 9: Open Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 4. Usage Instructions

### Mode A: Live Camera
1. Select your camera device index (`Camera 0`, `Camera 1`, etc.) from the dropdown.
2. Click **Start Camera**.
3. The system begins continuous video acquisition, vehicle detection, tracking, unique counting, plate recognition, and allocation verification.
4. Click **Stop** to release the camera.

### Mode B: Video or Image File Processing
1. Click **Choose File** and select an image (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) or video (`.mp4`, `.avi`, `.mov`, `.mkv`) file.
2. Click **Process Media**.
3. For images: The system instantly runs vehicle detection, plate localization, OCR, database verification, updates the display and table, and prints the terminal report.
4. For videos: The progress bar reflects real-time completion percentage until reaching the final frame.
5. In both cases, the dashboard remains active for subsequent runs.

---

## 5. Terminal Output Format

### Event 1: New Vehicle Detected (Immediate 1:1 Record Creation)
```
==================================================
NEW VEHICLE DETECTED
Tracking ID:           17
Vehicle Type:          CAR
Vehicle Confidence:    94.6%
Current Vehicle Count: 43
Database Series:       50
OCR Status:            PROCESSING
==================================================
```

### Event 2: OCR Confirmed Result
```
==================================================
OCR RESULT
Tracking ID:        17
Database Series:    50
Plate:              MH-12-AR-1234
Plate Confidence:   84.2%
OCR Status:         CONFIRMED
Parking Allocation: YES
Parking Slot:       A-12
==================================================
```

### Event 3: OCR Unclear Result
```
==================================================
OCR RESULT
Tracking ID:        18
Database Series:    51
Plate:              Not clear
OCR Status:         NOT_CLEAR
Parking Allocation: NO
==================================================
```

---

## 6. Indian License Plate Validation & Synchronization

- **Validation Rules**: Standard state code prefixes (e.g. `MH`, `DL`, `KA`, `HR`, `TN`, `UP`, `WB`), RTO digits, series letters, and registration numbers (e.g., `MH-12-AR-1234`), plus Bharat series (e.g. `22-BH-1234-AA`).
- **Unclear Handling**: Random OCR noise or non-matching patterns are strictly categorized as `Not clear` (never fabricated).
- **1:1 Data Pipeline**: Every unique tracked vehicle creates an immediate MySQL record. Asynchronous background OCR workers update the exact record upon completion.
- **Table `incoming_vehicles`**: Contains `series_number`, `tracking_id`, `vehicle_number`, `parking_allocation`, `parking_slot`, `ocr_status`, `plate_confidence`, `vehicle_type`, `detected_at`.
