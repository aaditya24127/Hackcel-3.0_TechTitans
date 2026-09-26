"""
Database Management Module
Handles MySQL connectivity, booking verification, incoming vehicle logging,
dynamic record creation, asynchronous OCR updates, and database synchronization.
"""

import os
import re
import sys
import logging
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger setup
logger = logging.getLogger("parking_database")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [DB]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Database Configuration from Environment
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "parking_system")

def get_connection_config():
    """Returns the database connection configuration dictionary."""
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": DB_NAME,
        "charset": "utf8mb4",
        "autocommit": True
    }

def get_db_connection():
    """
    Establishes and returns a fresh MySQL database connection.
    Returns None if the connection fails.
    """
    try:
        connection = mysql.connector.connect(**get_connection_config())
        if connection.is_connected():
            return connection
    except Error as e:
        logger.error(f"MySQL connection error: {e}")
        return None
    except Exception as ex:
        logger.error(f"Unexpected error while connecting to MySQL: {ex}")
        return None

def test_connection():
    """Tests if MySQL database can be connected to."""
    conn = get_db_connection()
    if conn:
        conn.close()
        return True, "Database connection successful."
    return False, "Unable to connect to MySQL database. Check your credentials and server status."

def init_db_schema():
    """
    Ensures that required columns exist in incoming_vehicles without dropping or losing existing data.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("Could not verify DB schema: MySQL is currently unreachable.")
        return False

    try:
        cursor = conn.cursor()
        # Check existing columns in incoming_vehicles
        cursor.execute("DESCRIBE incoming_vehicles")
        columns = [row[0].lower() for row in cursor.fetchall()]

        # Safely add any missing extended columns
        if "tracking_id" not in columns:
            cursor.execute("ALTER TABLE incoming_vehicles ADD COLUMN tracking_id INT NULL AFTER series_number")
            logger.info("Added 'tracking_id' column to incoming_vehicles.")

        if "parking_slot" not in columns:
            cursor.execute("ALTER TABLE incoming_vehicles ADD COLUMN parking_slot VARCHAR(20) DEFAULT '-' AFTER parking_allocation")
            logger.info("Added 'parking_slot' column to incoming_vehicles.")

        if "ocr_status" not in columns:
            cursor.execute("ALTER TABLE incoming_vehicles ADD COLUMN ocr_status VARCHAR(20) DEFAULT 'PROCESSING' AFTER parking_slot")
            logger.info("Added 'ocr_status' column to incoming_vehicles.")

        if "plate_confidence" not in columns:
            cursor.execute("ALTER TABLE incoming_vehicles ADD COLUMN plate_confidence FLOAT DEFAULT 0.0 AFTER ocr_status")
            logger.info("Added 'plate_confidence' column to incoming_vehicles.")

        if "vehicle_type" not in columns:
            cursor.execute("ALTER TABLE incoming_vehicles ADD COLUMN vehicle_type VARCHAR(20) DEFAULT 'CAR' AFTER plate_confidence")
            logger.info("Added 'vehicle_type' column to incoming_vehicles.")

        cursor.close()
        conn.close()
        return True
    except Error as e:
        logger.error(f"Error checking/updating DB schema: {e}")
        if conn and conn.is_connected():
            conn.close()
        return False

def format_plate_for_display(plate_str: str) -> str:
    """
    Converts a normalized plate string to standardized display format:
    e.g. MH12AB1234 -> MH-12-AB-1234
    MH12AB123 -> MH-12-AB-123
    KA01XX9999 -> KA-01-XX-9999
    22BH1234AA -> 22-BH-1234-AA
    If Not clear or empty, returns "Not clear".
    """
    if not plate_str or plate_str.strip().lower() in ("not clear", "not_clear", "unknown", "none", "-"):
        return "Not clear"

    clean = re.sub(r"[^A-Z0-9]", "", plate_str.upper().strip())
    if not clean or len(clean) < 4:
        return clean if clean else "Not clear"
    
    # 1. Bharat Series format: 2 Digits (Year) + BH + 1-4 Digits + 1-2 Letters
    m_bh = re.match(r"^(\d{2})BH(\d{1,4})([A-Z]{1,2})$", clean)
    if m_bh:
        yr, num, letters = m_bh.groups()
        return f"{yr}-BH-{num}-{letters}"

    # 2. Standard format: 2 State Letters + 1-2 Digits + 1-3 Series Letters + 1-4 Digits
    m_std = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$", clean)
    if m_std:
        state, rto, series, num = m_std.groups()
        return f"{state}-{rto.zfill(2)}-{series}-{num}"

    # 3. Format without series letters (older vehicle format): 2 Letters + 1-2 Digits + 1-4 Digits
    m_old = re.match(r"^([A-Z]{2})(\d{1,2})(\d{1,4})$", clean)
    if m_old:
        state, rto, num = m_old.groups()
        return f"{state}-{rto.zfill(2)}-{num}"

    # 4. Known 2-letter state prefix + rest
    valid_states = {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
        "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
        "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
        "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "UT", "WB"
    }
    if len(clean) >= 4 and clean[:2] in valid_states:
        return f"{clean[:2]}-{clean[2:]}"

    return clean

def check_booking(vehicle_number: str):
    """
    Checks whether a vehicle number exists in previous_bookings.
    Tolerates both raw strings (MH12AB1234) and formatted strings (MH-12-AB-1234).
    
    Returns:
        dict: {
            "is_booked": bool,
            "parking_allocation": "YES" | "NO",
            "parking_slot": str,
            "booking_status": bool
        }
    """
    if not vehicle_number or vehicle_number.strip().lower() in ("not clear", "not_clear", "-"):
        return {
            "is_booked": False,
            "parking_allocation": "NO",
            "parking_slot": "-",
            "booking_status": False
        }

    raw_clean = re.sub(r"[^A-Z0-9]", "", vehicle_number.upper().strip())
    conn = get_db_connection()
    
    if not conn:
        logger.warning(f"Database unavailable for booking verification of {raw_clean}.")
        return {
            "is_booked": False,
            "parking_allocation": "NO",
            "parking_slot": "-",
            "booking_status": False,
            "db_error": True
        }

    try:
        cursor = conn.cursor(dictionary=True)
        # Search by exact or clean alphanumeric comparison
        query = """
            SELECT vehicle_number, parking_slot, booking_status 
            FROM previous_bookings 
            WHERE REPLACE(REPLACE(UPPER(vehicle_number), '-', ''), ' ', '') = %s 
            LIMIT 1
        """
        cursor.execute(query, (raw_clean,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result.get("booking_status", True):
            return {
                "is_booked": True,
                "parking_allocation": "YES",
                "parking_slot": result.get("parking_slot") or "-",
                "booking_status": bool(result.get("booking_status"))
            }
        else:
            return {
                "is_booked": False,
                "parking_allocation": "NO",
                "parking_slot": "-",
                "booking_status": False
            }
    except Error as e:
        logger.error(f"Error querying previous_bookings: {e}")
        if conn and conn.is_connected():
            conn.close()
        return {
            "is_booked": False,
            "parking_allocation": "NO",
            "parking_slot": "-",
            "booking_status": False,
            "db_error": True
        }

def create_incoming_vehicle(tracking_id: int, vehicle_type: str = "CAR") -> int | None:
    """
    Immediately creates a database record when a new unique vehicle tracking ID is detected.
    Initializes with 'Not clear' and 'PROCESSING'.
    
    Returns:
        int | None: The series_number of the newly inserted record.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning(f"Could not create incoming vehicle record for Tracking ID {tracking_id} (DB unavailable).")
        return None

    try:
        cursor = conn.cursor()
        query = """
            INSERT INTO incoming_vehicles 
            (tracking_id, vehicle_number, parking_allocation, parking_slot, ocr_status, plate_confidence, vehicle_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            tracking_id,
            "Not clear",
            "NO",
            "-",
            "PROCESSING",
            0.0,
            vehicle_type.upper()
        ))
        series_number = cursor.lastrowid
        cursor.close()
        conn.close()
        logger.info(f"Created initial record #{series_number} for Tracking ID: {tracking_id}")
        return series_number
    except Error as e:
        # If columns do not exist yet in table, fallback to legacy schema insert
        try:
            query_fallback = "INSERT INTO incoming_vehicles (vehicle_number, parking_allocation) VALUES (%s, %s)"
            cursor.execute(query_fallback, ("Not clear", "NO"))
            series_number = cursor.lastrowid
            cursor.close()
            conn.close()
            return series_number
        except Exception:
            logger.error(f"Error creating initial incoming_vehicle record: {e}")
            if conn and conn.is_connected():
                conn.close()
            return None

def update_incoming_vehicle(
    series_number: int,
    vehicle_number: str,
    parking_allocation: str,
    parking_slot: str = "-",
    ocr_status: str = "CONFIRMED",
    plate_confidence: float = 0.0
) -> bool:
    """
    Updates an existing incoming vehicle record with verified OCR & booking results.
    """
    if not series_number:
        return False

    cleaned_number = vehicle_number.strip() if vehicle_number else "Not clear"
    allocation = "YES" if str(parking_allocation).upper() == "YES" else "NO"
    slot = parking_slot if allocation == "YES" else "-"

    conn = get_db_connection()
    if not conn:
        logger.warning(f"Could not update record #{series_number} (DB unavailable).")
        return False

    try:
        cursor = conn.cursor()
        query = """
            UPDATE incoming_vehicles 
            SET vehicle_number = %s,
                parking_allocation = %s,
                parking_slot = %s,
                ocr_status = %s,
                plate_confidence = %s
            WHERE series_number = %s
        """
        cursor.execute(query, (cleaned_number, allocation, slot, ocr_status, float(plate_confidence), series_number))
        cursor.close()
        conn.close()
        logger.info(f"Updated record #{series_number}: Plate='{cleaned_number}', Alloc={allocation}, Status={ocr_status}")
        return True
    except Error as e:
        # Fallback update if extended columns are not present
        try:
            cursor = conn.cursor()
            query_fallback = "UPDATE incoming_vehicles SET vehicle_number = %s, parking_allocation = %s WHERE series_number = %s"
            cursor.execute(query_fallback, (cleaned_number, allocation, series_number))
            cursor.close()
            conn.close()
            return True
        except Exception:
            logger.error(f"Error updating incoming_vehicle record #{series_number}: {e}")
            if conn and conn.is_connected():
                conn.close()
            return False

def get_incoming_vehicles(limit: int = 100):
    """
    Fetches the history of incoming vehicles detected by the system with formatted display values.
    
    Args:
        limit (int): Maximum number of records to return.
        
    Returns:
        list[dict]: List of incoming vehicle dictionaries.
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        # Check available columns
        query = """
            SELECT *
            FROM incoming_vehicles 
            ORDER BY series_number DESC 
            LIMIT %s
        """
        cursor.execute(query, (int(limit),))
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        # Format rows for clean UI presentation
        for row in results:
            raw_plate = row.get("vehicle_number", "Not clear")
            row["display_vehicle_number"] = format_plate_for_display(raw_plate)
            row["parking_slot"] = row.get("parking_slot") or ("-" if row.get("parking_allocation") != "YES" else "-")
            row["ocr_status"] = row.get("ocr_status") or ("CONFIRMED" if raw_plate != "Not clear" else "NOT_CLEAR")
            row["detected_at"] = str(row.get("detected_at", ""))
        return results
    except Error as e:
        logger.error(f"Error fetching incoming_vehicles history: {e}")
        if conn and conn.is_connected():
            conn.close()
        return []

def is_recently_recorded(vehicle_number: str, time_window_seconds: int = 60) -> bool:
    """Checks whether the specified vehicle number was logged within time window."""
    if not vehicle_number or vehicle_number == "Not clear":
        return False

    cleaned_number = re.sub(r"[^A-Z0-9]", "", vehicle_number.upper().strip())
    conn = get_db_connection()
    if not conn:
        return False

    try:
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT series_number FROM incoming_vehicles 
            WHERE REPLACE(REPLACE(UPPER(vehicle_number), '-', ''), ' ', '') = %s 
              AND detected_at >= NOW() - INTERVAL %s SECOND 
            LIMIT 1
        """
        cursor.execute(query, (cleaned_number, time_window_seconds))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Error as e:
        logger.error(f"Error checking recent duplicate: {e}")
        if conn and conn.is_connected():
            conn.close()
        return False

# Initialize schema migration on module load
init_db_schema()
