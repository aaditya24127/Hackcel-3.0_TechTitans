"""
Optical Character Recognition (OCR) & Plate Validation Module
Handles multi-pass image enhancement, upscaling, perspective alignment,
PaddleOCR with EasyOCR fallback, positional character confusion resolution,
strict Indian registration plate validation, and debug image logging.
"""

import os
import re
import sys
import time
import logging
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"

# Logger setup
logger = logging.getLogger("parking_ocr")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [OCR]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Configuration from Environment
OCR_CONF_THRESH = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.55"))
PLATE_CONF_THRESH = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.35"))
DEBUG_PLATE_IMAGES = os.getenv("DEBUG_PLATE_IMAGES", "True").lower() in ("true", "1", "yes")
PLATE_DEBUG = os.getenv("PLATE_DEBUG", "True").lower() in ("true", "1", "yes")

# Valid Indian States & Union Territories
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "UT", "WB"
}

# Character Confusion Maps (Digit <-> Letter)
CHAR_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6'
}

CHAR_TO_LETTER = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G'
}

# Ensure debug directories exist if enabled
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
DEBUG_PLATES_DIR = os.path.join(DEBUG_DIR, "plates")
DEBUG_ENHANCED_DIR = os.path.join(DEBUG_DIR, "enhanced")
DEBUG_FAILED_DIR = os.path.join(DEBUG_DIR, "failed")

if DEBUG_PLATE_IMAGES or PLATE_DEBUG:
    os.makedirs(DEBUG_PLATES_DIR, exist_ok=True)
    os.makedirs(DEBUG_ENHANCED_DIR, exist_ok=True)
    os.makedirs(DEBUG_FAILED_DIR, exist_ok=True)


class IndianPlateValidator:
    """
    Normalizes and validates vehicle registration numbers with smart formatting
    for Indian plates, while flexibly accepting any readable registration number.
    """
    @staticmethod
    def clean_raw_string(text: str) -> str:
        """Removes all whitespace, hyphens, and non-alphanumerics, returning uppercase."""
        if not text:
            return ""
        return re.sub(r"[^A-Z0-9]", "", str(text).upper().strip())

    @classmethod
    def apply_positional_corrections(cls, candidate: str) -> str:
        """
        Applies structure-aware digit/letter correction if the plate starts with a known Indian state code:
        Format: [State 2 letters][RTO 2 digits][Series 1-3 letters][Number 1-4 digits]
        """
        if len(candidate) < 5:
            return candidate

        # If candidate already matches a clean Indian plate format, preserve exact characters
        if re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$", candidate) or \
           re.match(r"^(\d{2})BH(\d{1,4})([A-Z]{1,2})$", candidate):
            return candidate

        chars = list(candidate)
        n = len(chars)

        # Positions 0 and 1: State Code check
        test_state = [CHAR_TO_LETTER.get(chars[0], chars[0]), CHAR_TO_LETTER.get(chars[1], chars[1])]
        if "".join(test_state) in INDIAN_STATE_CODES:
            chars[0], chars[1] = test_state[0], test_state[1]

            # Positions 2 and 3: RTO code
            if n >= 3 and chars[2] in CHAR_TO_DIGIT:
                chars[2] = CHAR_TO_DIGIT[chars[2]]
            if n >= 4 and chars[3] in CHAR_TO_DIGIT:
                chars[3] = CHAR_TO_DIGIT[chars[3]]

            # Trailing characters fix: convert trailing letter noise (O, I, Z, S, B) to digits at the end
            for i in range(n - 1, max(3, n - 4), -1):
                if chars[i] in CHAR_TO_DIGIT:
                    chars[i] = CHAR_TO_DIGIT[chars[i]]

        return "".join(chars)

    @classmethod
    def format_for_display(cls, text: str) -> str:
        """
        Formats any registration number cleanly, applying standard Indian hyphenation where applicable:
        e.g. MH12AB1234 -> MH-12-AB-1234
        MH12AB123 -> MH-12-AB-123
        KA01XX9999 -> KA-01-XX-9999
        KL41L701 -> KL-41-L-701
        22BH1234AA -> 22-BH-1234-AA
        """
        clean = cls.clean_raw_string(text)
        if not clean or clean.lower() in ("notclear", "not_clear", "none", "unknown"):
            return "Not clear"

        # 1. Bharat Series (BH): [Year 2D] BH [Number 1-4D] [Series 1-2L]
        m_bh = re.match(r"^(\d{2})BH(\d{1,4})([A-Z]{1,2})$", clean)
        if m_bh:
            yr, num, letters = m_bh.groups()
            return f"{yr}-BH-{num}-{letters}"

        # 2. Standard Indian Plate: [State 2L][RTO 1-2D][Series 1-3L][Number 1-4D]
        # e.g., MH12AB1234 -> MH-12-AB-1234, MH12AB123 -> MH-12-AB-123, KA01XX9999 -> KA-01-XX-9999
        m_std = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})$", clean)
        if m_std:
            state, rto, series, num = m_std.groups()
            return f"{state}-{rto.zfill(2)}-{series}-{num}"

        # 3. Older/Variant Format: [State 2L][RTO 1-2D][Number 1-4D]
        # e.g., MH121234 -> MH-12-1234
        m_old = re.match(r"^([A-Z]{2})(\d{1,2})(\d{1,4})$", clean)
        if m_old:
            state, rto, num = m_old.groups()
            return f"{state}-{rto.zfill(2)}-{num}"

        # 4. Known State prefix with remaining alphanumeric string: [State 2L][Rest]
        if len(clean) >= 4 and clean[:2] in INDIAN_STATE_CODES:
            return f"{clean[:2]}-{clean[2:]}"

        # 5. Default readable string
        return clean

    @classmethod
    def validate_and_parse(cls, raw_text: str, confidence: float) -> tuple[bool, str, str]:
        """
        Accepts any reasonably readable vehicle registration number if confidence is good.
        Normalizes to clean alphanumeric and formatted display.
        
        Returns:
            tuple: (is_valid: bool, normalized_raw: str, formatted_display: str)
        """
        if confidence < OCR_CONF_THRESH or not raw_text:
            return False, "Not clear", "Not clear"

        clean = cls.clean_raw_string(raw_text)

        # Minimum usable plate length: 4 to 14 characters
        if len(clean) < 4 or len(clean) > 14:
            return False, "Not clear", "Not clear"

        # Discard pure noise / single character repetition (e.g. '----', '0000', 'XXXX')
        if len(set(clean)) <= 1 and len(clean) < 6:
            return False, "Not clear", "Not clear"

        # Apply positional corrections if applicable
        corrected = cls.apply_positional_corrections(clean)

        # Format for display
        display_str = cls.format_for_display(corrected)

        return True, corrected, display_str


class OCREngine:
    """
    Manages OCR model initialization (PaddleOCR with EasyOCR fallback),
    multi-pass image preprocessing variants, and OCR inference.
    """
    def __init__(self):
        self.engine_name = "NONE"
        self.paddle_ocr = None
        self.easy_ocr = None
        self._init_ocr()

    def _init_ocr(self):
        """Initializes PaddleOCR with EasyOCR as a clean fallback."""
        # Ensure torch is imported first to avoid Windows DLL collision
        try:
            import torch
        except Exception:
            pass

        # 1. Try initializing PaddleOCR
        try:
            from paddleocr import PaddleOCR
            logger.info("Attempting to initialize PaddleOCR (English)...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
            self.engine_name = "PaddleOCR"
            logger.info("OCR ENGINE: PaddleOCR initialized successfully.")
            return
        except Exception as e:
            logger.warning(f"PaddleOCR not available or failed initialization ({e}). Falling back to EasyOCR.")
            self.paddle_ocr = None

        # 2. Fallback: Initialize EasyOCR
        self._init_easyocr_fallback()

    def _init_easyocr_fallback(self):
        """Initializes EasyOCR fallback reader."""
        try:
            import easyocr
            import torch
            has_gpu = torch.cuda.is_available()
            logger.info(f"Initializing EasyOCR (English, GPU={has_gpu})...")
            self.easy_ocr = easyocr.Reader(['en'], gpu=has_gpu, verbose=False)
            self.engine_name = "EasyOCR FALLBACK"
            logger.info("OCR ENGINE: EasyOCR FALLBACK initialized successfully.")
        except Exception as ex:
            logger.error(f"Failed to initialize EasyOCR fallback: {ex}")
            self.easy_ocr = None
            self.engine_name = "NONE"

    @staticmethod
    def generate_image_variants(crop_img: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
        Generates multi-pass enhanced and upscaled variants of the plate crop.
        """
        if crop_img is None or crop_img.size == 0:
            return []

        h, w = crop_img.shape[:2]
        # Determine upscale multiplier based on input height
        if h < 35:
            scale = 3.5
        elif h < 60:
            scale = 2.5
        elif h < 100:
            scale = 1.8
        else:
            scale = 1.2

        # 1. High-quality Cubic Upscale
        enlarged_bgr = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Contrast enhancement via CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 3. Bilateral Filter Denoising (preserves text edges)
        denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)

        # 4. Sharpening Filter
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)

        # 5. Adaptive Gaussian Thresholding
        adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 19, 5)

        # 6. Otsu Thresholding
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return [
            ("raw_enlarged", enlarged_bgr),
            ("denoised_clahe", denoised),
            ("sharpened", sharpened),
            ("adaptive_threshold", adaptive),
            ("otsu_threshold", otsu)
        ]

    def _run_paddle_ocr(self, img: np.ndarray) -> list[tuple[str, float]]:
        """Runs PaddleOCR on an image variant, extracting text blocks and joined combinations."""
        if self.paddle_ocr is None:
            return []
        try:
            if len(img.shape) == 2:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            res = self.paddle_ocr.ocr(img_rgb)
            extracted = []
            if res and len(res) > 0 and res[0] is not None:
                for line in res[0]:
                    if len(line) >= 2 and isinstance(line[1], (tuple, list)):
                        txt, conf = str(line[1][0]), float(line[1][1])
                        # Get y coordinate of bounding box for top-down sorting
                        box = line[0]
                        y_pos = box[0][1] if box and len(box) > 0 else 0
                        x_pos = box[0][0] if box and len(box) > 0 else 0
                        extracted.append((y_pos, x_pos, txt, conf))

            if not extracted:
                return []

            # Sort top-to-bottom, then left-to-right
            extracted.sort(key=lambda item: (item[0], item[1]))

            candidates = []
            # 1. Individual text lines
            for _, _, txt, conf in extracted:
                candidates.append((txt, conf))

            # 2. Joined text lines (for multi-line Indian plates like MH12 / AR1234)
            if len(extracted) > 1:
                joined_text = " ".join(t[2] for t in extracted)
                avg_conf = sum(t[3] for t in extracted) / len(extracted)
                candidates.append((joined_text, avg_conf))

            return candidates
        except Exception as e:
            logger.debug(f"PaddleOCR runtime error ({e}). Switching to EasyOCR fallback.")
            self.paddle_ocr = None
            if self.easy_ocr is None:
                self._init_easyocr_fallback()
            return self._run_easy_ocr(img)

    def _run_easy_ocr(self, img: np.ndarray) -> list[tuple[str, float]]:
        """Runs EasyOCR on an image variant, extracting text blocks and joined combinations."""
        if self.easy_ocr is None:
            return []
        try:
            results = self.easy_ocr.readtext(
                img,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -."
            )
            if not results:
                return []

            # results: list of (bbox, text, conf)
            extracted = []
            for item in results:
                box, txt, conf = item[0], str(item[1]), float(item[2])
                y_pos = box[0][1] if box and len(box) > 0 else 0
                x_pos = box[0][0] if box and len(box) > 0 else 0
                extracted.append((y_pos, x_pos, txt, conf))

            extracted.sort(key=lambda item: (item[0], item[1]))

            candidates = []
            # 1. Individual text lines
            for _, _, txt, conf in extracted:
                candidates.append((txt, conf))

            # 2. Joined text lines (for multi-line Indian plates like MH12 / AR1234)
            if len(extracted) > 1:
                joined_text = "".join(t[2] for t in extracted)
                avg_conf = sum(t[3] for t in extracted) / len(extracted)
                candidates.append((joined_text, avg_conf))

            return candidates
        except Exception as e:
            logger.debug(f"EasyOCR inference error: {e}")
            return []

    def recognize_plate(self, plate_crop: np.ndarray, tracking_id: int = None) -> tuple[bool, str, str, float]:
        """
        Executes multi-pass OCR on multiple enhanced variants of the plate crop,
        validates results against Indian registration standards, and returns the top candidate.
        
        Returns:
            tuple: (is_valid: bool, normalized_raw: str, formatted_display: str, confidence: float)
        """
        if plate_crop is None or plate_crop.size == 0:
            return False, "Not clear", "Not clear", 0.0

        variants = self.generate_image_variants(plate_crop)
        best_valid = False
        best_raw = "Not clear"
        best_display = "Not clear"
        best_conf = 0.0
        best_variant_img = None
        all_candidate_logs = []

        for variant_name, img_var in variants:
            # Run active OCR engine
            if self.engine_name == "PaddleOCR" and self.paddle_ocr is not None:
                ocr_results = self._run_paddle_ocr(img_var)
            else:
                ocr_results = self._run_easy_ocr(img_var)

            for raw_text, conf in ocr_results:
                is_valid, norm_raw, fmt_display = IndianPlateValidator.validate_and_parse(raw_text, conf)
                all_candidate_logs.append((raw_text, conf, is_valid, fmt_display))

                if is_valid and conf > best_conf:
                    best_valid = True
                    best_raw = norm_raw
                    best_display = fmt_display
                    best_conf = conf
                    best_variant_img = img_var

            # Early exit if a high-confidence validated plate has been obtained
            if best_valid and best_conf >= 0.75:
                break

        # Save debug images if enabled
        if DEBUG_PLATE_IMAGES and tracking_id is not None:
            ts = int(time.time() * 1000) % 1000000
            try:
                cv2.imwrite(os.path.join(DEBUG_PLATES_DIR, f"tracking_{tracking_id}_{ts}.jpg"), plate_crop)
                if best_valid and best_variant_img is not None:
                    cv2.imwrite(os.path.join(DEBUG_ENHANCED_DIR, f"tracking_{tracking_id}_{ts}.jpg"), best_variant_img)
                elif not best_valid:
                    cv2.imwrite(os.path.join(DEBUG_FAILED_DIR, f"tracking_{tracking_id}_{ts}.jpg"), plate_crop)
            except Exception:
                pass

        return best_valid, best_raw, best_display, best_conf
