"""
OCR Pipeline for AFCAT Exam Papers
==================================
Extracts text from PDFs and scanned images using PaddleOCR/EasyOCR.
Handles multi-column layouts, preprocessing, and noise removal.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Available OCR engines."""
    PADDLE = "paddle"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


@dataclass
class OCRResult:
    """Single OCR detection result."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    page_num: int = 0


@dataclass
class NeighborContext:
    """Context inferred from neighboring questions for gap-filling."""
    prev_question: Optional['ExtractedQuestion'] = None
    next_question: Optional['ExtractedQuestion'] = None
    prev_topic_hint: str = ""
    next_topic_hint: str = ""
    inferred_section: str = "unknown"
    confidence: float = 0.0


@dataclass
class ExtractedQuestion:
    """Extracted MCQ question with options."""
    question_number: int
    text: str
    options: List[str]
    raw_bbox: Optional[Tuple] = None
    confidence: float = 0.0
    page_num: int = 0
    question_type: str = "unknown"
    has_diagram_reference: bool = False
    neighbor_context: Optional[NeighborContext] = None


class ImagePreprocessor:
    """
    Preprocessing utilities for better OCR accuracy on exam papers.
    Handles deskewing, denoising, contrast enhancement, etc.
    """
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Fix rotation/skew in scanned documents."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            coords = np.column_stack(np.where(gray > 0))
            
            if len(coords) < 10:
                return image
                
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if abs(angle) > 0.5:  # Only rotate if significant skew
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
            return image
        except Exception as e:
            logger.warning(f"Deskew failed: {e}")
            return image
    
    @staticmethod
    def remove_noise(image: np.ndarray, strength: str = "medium") -> np.ndarray:
        """Remove scanner noise and artifacts using bilateral filter."""
        try:
            if strength == "light":
                denoised = cv2.bilateralFilter(image, 5, 50, 50)
            elif strength == "strong":
                denoised = cv2.bilateralFilter(image, 15, 100, 100)
                # Additional morphological cleanup for heavy noise
                kernel = np.ones((2, 2), np.uint8)
                denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
            else:  # medium
                denoised = cv2.bilateralFilter(image, 9, 75, 75)
            return denoised
        except Exception:
            return image
    
    @staticmethod
    def sharpen(image: np.ndarray) -> np.ndarray:
        """Sharpen text edges for better OCR recognition."""
        try:
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            return sharpened
        except Exception:
            return image
    
    @staticmethod
    def remove_borders(image: np.ndarray, border_px: int = 10) -> np.ndarray:
        """Remove black borders from scanned pages."""
        try:
            h, w = image.shape[:2]
            if h > 2 * border_px and w > 2 * border_px:
                return image[border_px:h-border_px, border_px:w-border_px]
            return image
        except Exception:
            return image
    
    @staticmethod
    def normalize_illumination(image: np.ndarray) -> np.ndarray:
        """Correct uneven lighting from scanned documents."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Estimate background using morphological opening
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Divide original by background to normalize
            normalized = cv2.divide(gray, background, scale=255)
            
            if len(image.shape) == 3:
                return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
            return normalized
        except Exception:
            return image
    
    @staticmethod
    def binarize(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
        """Convert to black and white for better OCR."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        if method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
        return binary
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        try:
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                lab = cv2.merge([l, a, b])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except Exception:
            return image
    
    @staticmethod
    def resize_for_ocr(image: np.ndarray, max_dimension: int = 4000) -> np.ndarray:
        """Resize large images to prevent memory issues."""
        h, w = image.shape[:2]
        
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        return image
    
    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        deskew: bool = True,
        denoise: bool = True,
        binarize: bool = False,
        enhance: bool = True,
        resize: bool = True,
        quality_level: str = "standard"
    ) -> np.ndarray:
        """Full preprocessing pipeline for exam paper images.
        
        Args:
            quality_level: "fast", "standard", or "max_quality"
                - fast: minimal processing for clean digital PDFs
                - standard: balanced processing (default)
                - max_quality: aggressive processing for difficult scans
        """
        result = image.copy()
        
        if quality_level == "fast":
            # Minimal processing for clean PDFs
            if resize:
                result = self.resize_for_ocr(result)
            return result
        
        if resize:
            result = self.resize_for_ocr(result)
        
        if quality_level == "max_quality":
            # Aggressive pipeline for difficult scans
            result = self.remove_borders(result)
            result = self.normalize_illumination(result)
            result = self.enhance_contrast(result)
            result = self.remove_noise(result, strength="strong")
            if deskew:
                result = self.deskew(result)
            result = self.sharpen(result)
            # For max quality, always binarize
            result = self.binarize(result, method="adaptive")
        else:
            # Standard processing
            if enhance:
                result = self.enhance_contrast(result)
            if denoise:
                result = self.remove_noise(result, strength="medium")
            if deskew:
                result = self.deskew(result)
            if binarize:
                result = self.binarize(result)
            
        return result
    
    def multi_pass_preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple preprocessed versions for OCR ensemble.
        
        Returns list of images processed with different settings to maximize
        text extraction from challenging scans.
        """
        versions = []
        
        # Version 1: Standard processing
        v1 = self.preprocess_for_ocr(image, quality_level="standard")
        versions.append(("standard", v1))
        
        # Version 2: Max quality with binarization
        v2 = self.preprocess_for_ocr(image, quality_level="max_quality")
        versions.append(("max_quality", v2))
        
        # Version 3: High contrast + Otsu binarization
        v3 = self.enhance_contrast(image.copy())
        v3 = self.binarize(v3, method="otsu")
        versions.append(("otsu", v3))
        
        return versions


class ExamPaperOCR:
    """
    Main OCR class for extracting text from AFCAT exam papers.
    Supports PDF, scanned images, and multi-column layouts.
    Uses PaddleOCR (primary) or EasyOCR (fallback).
    """
    
    def __init__(
        self,
        engine: OCREngine = OCREngine.EASYOCR,  # EasyOCR as default (easier install)
        languages: List[str] = ["en"],
        gpu: bool = False,
        quality_level: str = "standard"  # "fast", "standard", "max_quality"
    ):
        self.engine = engine
        self.languages = languages
        self.gpu = gpu
        self.quality_level = quality_level
        self.preprocessor = ImagePreprocessor()
        self._ocr_model = None
        self._pdf_handler = None
        self.page_images: Dict[int, List[Dict]] = {}
        self._initialize_engine()
        
    def _initialize_engine(self):
        """Initialize the selected OCR engine."""
        try:
            if self.engine == OCREngine.PADDLE:
                self._init_paddle()
            elif self.engine == OCREngine.EASYOCR:
                self._init_easyocr()
            elif self.engine == OCREngine.TESSERACT:
                self._init_tesseract()
        except ImportError as e:
            logger.warning(f"Failed to initialize {self.engine.value}: {e}")
            # Try fallback engines
            if self.engine != OCREngine.EASYOCR:
                try:
                    self._init_easyocr()
                    self.engine = OCREngine.EASYOCR
                    logger.info("Fallback to EasyOCR")
                except ImportError:
                    pass
                    
    def _init_paddle(self):
        """Initialize PaddleOCR."""
        from paddleocr import PaddleOCR
        self._ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=self.gpu,
            show_log=False
        )
        logger.info("Initialized PaddleOCR engine")
        
    def _init_easyocr(self):
        """Initialize EasyOCR."""
        import easyocr
        self._ocr_model = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            verbose=False
        )
        logger.info("Initialized EasyOCR engine")
        
    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        import pytesseract
        self._ocr_model = pytesseract
        logger.info("Initialized Tesseract engine")
        
    def extract_from_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        dpi: Optional[int] = None  # Auto-select based on quality_level
    ) -> List[OCRResult]:
        """
        Extract text from PDF - tries native extraction first, then OCR.
        
        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers to process (0-indexed), None = all
            dpi: Resolution for rendering PDF pages as images (auto-selected if None)
            
        Returns:
            List of OCRResult objects
        """
        pdf_path = Path(pdf_path)
        results = []
        self.page_images = {}
        
        # Auto-select DPI based on quality level
        if dpi is None:
            if self.quality_level == "fast":
                dpi = 200
            elif self.quality_level == "max_quality":
                dpi = 400  # Higher DPI for difficult scans
            else:
                dpi = 300  # Standard
        
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install pymupdf")
            return results
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            pages_to_process = pages if pages else list(range(total_pages))
            
            for page_num in pages_to_process:
                if page_num >= total_pages:
                    continue
                    
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                # Capture embedded images for figure detection
                try:
                    images_on_page = []
                    for img in page.get_images(full=True):
                        xref = img[0]
                        bbox = page.get_image_bbox(xref)
                        images_on_page.append({
                            "xref": xref,
                            "bbox": (
                                int(bbox.x0), int(bbox.y0),
                                int(bbox.x1), int(bbox.y1)
                            ),
                            "width": int(img[2]),
                            "height": int(img[3])
                        })
                    if images_on_page:
                        self.page_images[page_num] = images_on_page
                except Exception:
                    # Image capture is best-effort; continue even if it fails
                    pass
                
                # Try native text extraction first (for digital PDFs)
                native_text = page.get_text("text")
                
                if len(native_text.strip()) > 100:
                    # Has substantial native text
                    results.append(OCRResult(
                        text=native_text,
                        confidence=1.0,
                        bbox=(0, 0, int(page.rect.width), int(page.rect.height)),
                        page_num=page_num
                    ))
                else:
                    # Fall back to OCR for scanned pages
                    pix = page.get_pixmap(dpi=dpi)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )
                    
                    # Convert RGBA to BGR if needed
                    if pix.n == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    elif pix.n == 1:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    # Apply preprocessing based on quality level
                    img = self.preprocessor.preprocess_for_ocr(
                        img, quality_level=self.quality_level
                    )
                        
                    page_results = self._ocr_image(img, page_num)
                    results.extend(page_results)
                    
            doc.close()
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise
            
        return results
    
    def extract_from_image(
        self,
        image_path: Union[str, Path, np.ndarray],
        preprocess: bool = True,
        page_num: int = 0
    ) -> List[OCRResult]:
        """
        Extract text from a single image file.
        
        Args:
            image_path: Path to image file or numpy array
            preprocess: Whether to apply preprocessing
            page_num: Page number for metadata
            
        Returns:
            List of OCRResult objects
        """
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            image = image_path
            
        if preprocess:
            image = self.preprocessor.preprocess_for_ocr(image)
            
        return self._ocr_image(image, page_num)
    
    def _ocr_image(self, image: np.ndarray, page_num: int = 0) -> List[OCRResult]:
        """Run OCR on an image using the selected engine."""
        results = []
        
        if self._ocr_model is None:
            logger.error("OCR engine not initialized")
            return results
        
        try:
            if self.engine == OCREngine.PADDLE:
                results = self._ocr_paddle(image, page_num)
            elif self.engine == OCREngine.EASYOCR:
                results = self._ocr_easyocr(image, page_num)
            elif self.engine == OCREngine.TESSERACT:
                results = self._ocr_tesseract(image, page_num)
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            
        return results
    
    def _ocr_paddle(self, image: np.ndarray, page_num: int) -> List[OCRResult]:
        """Run PaddleOCR on image."""
        results = []
        ocr_output = self._ocr_model.ocr(image, cls=True)
        
        if ocr_output and ocr_output[0]:
            for line in ocr_output[0]:
                bbox_points = line[0]
                text, confidence = line[1]
                
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                
                results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=tuple(int(v) for v in bbox),
                    page_num=page_num
                ))
                
        return results
    
    def _ocr_easyocr(self, image: np.ndarray, page_num: int) -> List[OCRResult]:
        """Run EasyOCR on image."""
        results = []
        ocr_output = self._ocr_model.readtext(image)
        
        for (bbox, text, confidence) in ocr_output:
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bbox_rect = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            results.append(OCRResult(
                text=text,
                confidence=confidence,
                bbox=tuple(int(v) for v in bbox_rect),
                page_num=page_num
            ))
            
        return results
    
    def _ocr_tesseract(self, image: np.ndarray, page_num: int) -> List[OCRResult]:
        """Run Tesseract OCR on image."""
        results = []
        data = self._ocr_model.image_to_data(
            image,
            lang='eng',
            output_type=self._ocr_model.Output.DICT
        )
        
        for i, text in enumerate(data['text']):
            if text.strip():
                conf = float(data['conf'][i])
                if conf > 0:  # Valid detection
                    results.append(OCRResult(
                        text=text,
                        confidence=conf / 100,
                        bbox=(
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ),
                        page_num=page_num
                    ))
                    
        return results
    
    def detect_columns(
        self,
        image: np.ndarray,
        min_column_width: int = 200
    ) -> List[Tuple[int, int]]:
        """
        Detect multi-column layout in exam papers.
        
        Returns:
            List of (x_start, x_end) for each column
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold and find vertical gaps
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Sum pixels vertically
        vertical_sum = np.sum(binary, axis=0)
        
        # Find gaps (low pixel density areas)
        threshold = np.mean(vertical_sum) * 0.1
        is_gap = vertical_sum < threshold
        
        # Find column boundaries
        columns = []
        in_column = False
        start = 0
        
        for i, gap in enumerate(is_gap):
            if not gap and not in_column:
                start = i
                in_column = True
            elif gap and in_column:
                if i - start >= min_column_width:
                    columns.append((start, i))
                in_column = False
                
        if in_column and image.shape[1] - start >= min_column_width:
            columns.append((start, image.shape[1]))
            
        return columns if columns else [(0, image.shape[1])]
    
    def extract_with_layout(
        self,
        image: np.ndarray,
        handle_columns: bool = True
    ) -> str:
        """Extract text respecting multi-column layout."""
        if handle_columns:
            columns = self.detect_columns(image)
        else:
            columns = [(0, image.shape[1])]
            
        all_text = []
        
        for col_start, col_end in columns:
            col_image = image[:, col_start:col_end]
            results = self._ocr_image(col_image)
            
            # Sort by vertical position
            results.sort(key=lambda r: r.bbox[1])
            
            col_text = '\n'.join(r.text for r in results)
            all_text.append(col_text)
            
        return '\n\n'.join(all_text)
    
    def get_full_text(self, ocr_results: List[OCRResult]) -> str:
        """Combine OCR results into full text, sorted by position."""
        # Sort by page, then y position, then x position
        sorted_results = sorted(
            ocr_results,
            key=lambda r: (r.page_num, r.bbox[1], r.bbox[0])
        )
        
        full_text = []
        current_page = -1
        
        for result in sorted_results:
            if result.page_num != current_page:
                if current_page >= 0:
                    full_text.append(f"\n\n--- PAGE {result.page_num + 1} ---\n\n")
                current_page = result.page_num
            full_text.append(result.text)
            
        return '\n'.join(full_text)

    def get_page_images(self) -> Dict[int, List[Dict]]:
        """Return captured page image metadata (bbox, xref)."""
        return self.page_images


class MCQExtractor:
    """
    Extracts structured MCQ questions from OCR text.
    Handles various AFCAT question paper formats including:
    - Official papers (digital PDFs)
    - Memory-based papers (varied formats)
    - Scanned papers with OCR artifacts
    """
    
    # Roman numeral mapping for conversion
    ROMAN_TO_INT = {
        'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5,
        'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10,
        'xi': 11, 'xii': 12, 'xiii': 13, 'xiv': 14, 'xv': 15,
        'xvi': 16, 'xvii': 17, 'xviii': 18, 'xix': 19, 'xx': 20
    }
    
    # Expanded patterns for Indian competitive exam questions
    QUESTION_PATTERNS = [
        # "1. Question text" or "1) Question text"
        r'(?:^|\n)\s*(?P<num>\d+)\s*[.)]\s*(?P<text>.+?)(?=(?:\n\s*\d+\s*[.)]|\s*\(A\)|\s*\(a\)|$))',
        # "Q.1 Question text" or "Q1. Question text" or "Q 1:" 
        r'(?:^|\n)\s*Q\.?\s*(?P<num>\d+)\s*[.:)]?\s*(?P<text>.+?)(?=(?:\n\s*Q\.?\s*\d+|$))',
        # "Question 1:" or "Question 1."
        r'(?:^|\n)\s*Question\s+(?P<num>\d+)\s*[.:)]?\s*(?P<text>.+?)(?=(?:\n\s*Question\s+\d+|$))',
        # "(1) Question text" - parenthesized numbers
        r'(?:^|\n)\s*\((?P<num>\d+)\)\s*(?P<text>.+?)(?=(?:\n\s*\(\d+\)|\s*\(A\)|\s*\(a\)|$))',
        # "[1] Question text" - bracketed numbers
        r'(?:^|\n)\s*\[(?P<num>\d+)\]\s*(?P<text>.+?)(?=(?:\n\s*\[\d+\]|\s*\(A\)|\s*\(a\)|$))',
        # "Ques 1:" or "Ques. 1:"
        r'(?:^|\n)\s*Ques\.?\s*(?P<num>\d+)\s*[.:)]?\s*(?P<text>.+?)(?=(?:\n\s*Ques\.?\s*\d+|$))',
        # Just numbered lines starting with capital letter
        r'(?:^|\n)\s*(?P<num>\d+)\s*\.\s*(?P<text>[A-Z].+?)(?=(?:\n\s*\d+\s*\.|\n\s*$|$))',
        # "1:" style (colon after number)
        r'(?:^|\n)\s*(?P<num>\d+)\s*:\s*(?P<text>.+?)(?=(?:\n\s*\d+\s*:|\s*\(A\)|\s*\(a\)|$))',
    ]
    
    # RELAXED MATH PATTERNS - catches bad OCR in Math section (Q81-100)
    # These are tried specifically for questions in the 80-100 range
    MATH_RELAXED_PATTERNS = [
        # "88 A train" - Missing dot after number (Q81-100 specific)
        r'(?:^|\n)\s*(?P<num>8[1-9]|9\d|100)\s+(?P<text>[A-Z].+?)(?=(?:\n\s*(?:8[1-9]|9\d|100)\s+[A-Z]|\s*\(A\)|\s*\(a\)|$))',
        # "Q88" or "Q.88" - Q prefix variations for Q81-100
        r'(?:^|\n)\s*Q\.?\s*(?P<num>8[1-9]|9\d|100)\s*(?P<text>.+?)(?=(?:\n\s*Q\.?\s*(?:8[1-9]|9\d|100)|\s*\(A\)|\s*\(a\)|$))',
        # "(88)" or "[88]" - Bracketed numbers for Q81-100
        r'(?:^|\n)\s*[\[\(](?P<num>8[1-9]|9\d|100)[\]\)]\s*(?P<text>.+?)(?=(?:\n\s*[\[\(](?:8[1-9]|9\d|100)[\]\)]|\s*\(A\)|\s*\(a\)|$))',
        # Just "88 " followed by any text - most permissive for Q81-100
        r'(?:^|\n)\s*(?P<num>8[1-9]|9\d|100)\s+(?P<text>.+?)(?=(?:\n\s*(?:8[1-9]|9\d|100)\s+|\s*\(A\)|\s*\(a\)|$))',
    ]
    
    # Roman numeral pattern (handled separately)
    ROMAN_PATTERN = r'(?:^|\n)\s*(?P<num>(?:x{0,3})(?:ix|iv|v?i{0,3}))\s*[.)]\s*(?P<text>.+?)(?=(?:\n\s*(?:x{0,3})(?:ix|iv|v?i{0,3})\s*[.)]|$))'
    
    # More flexible option patterns that handle newlines and various formats
    OPTION_PATTERNS = [
        # (A) option text (B) option text... - handles inline or multi-line
        r'\(([A-Da-d])\)\s*([^(]+?)(?=\s*\([A-Da-d]\)|$)',
        # A. option text B. option text...
        r'(?:^|\s)([A-Da-d])\.\s*([^A-Da-d.]+?)(?=\s*[A-Da-d]\.|$)',
        # [A] option text [B] option text...
        r'\[([A-Da-d])\]\s*([^\[]+?)(?=\s*\[[A-Da-d]\]|$)',
        # a) option text b) option text...
        r'(?:^|\s)([A-Da-d])\)\s*([^a-dA-D)]+?)(?=\s*[A-Da-d]\)|$)',
        # (1) (2) (3) (4) style options
        r'\(([1-4])\)\s*([^(]+?)(?=\s*\([1-4]\)|$)',
        # Options on separate lines: A option / B option
        r'(?:^|\n)\s*([A-Da-d])\s*[.:)]\s*(.+?)(?=\n\s*[A-Da-d]\s*[.:)]|\n\n|$)',
    ]
    
    def __init__(self):
        self.compiled_q_patterns = [
            re.compile(p, re.DOTALL | re.MULTILINE | re.IGNORECASE) 
            for p in self.QUESTION_PATTERNS
        ]
        self.compiled_opt_patterns = [
            re.compile(p, re.DOTALL | re.MULTILINE) 
            for p in self.OPTION_PATTERNS
        ]
        self.roman_pattern = re.compile(self.ROMAN_PATTERN, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        
        # Compile relaxed Math patterns for rescue operations
        self.compiled_math_patterns = [
            re.compile(p, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            for p in self.MATH_RELAXED_PATTERNS
        ]
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize OCR text to improve extraction accuracy.
        Handles common OCR artifacts and formatting issues.
        """
        # Fix common OCR errors
        ocr_fixes = {
            '|': 'I',           # Pipe often misread as I
            '0': 'O',           # In text context (we'll be careful)
            'l1': 'll',         # Common misread
            '1l': 'll',
            'rn': 'm',          # r+n often = m
            'vv': 'w',          # v+v often = w
            '\\': '/',          # Backslash to forward
        }
        
        normalized = text
        
        # Merge broken lines (hyphenation at line breaks)
        normalized = re.sub(r'-\s*\n\s*', '', normalized)
        
        # Merge lines that appear to be continuation (lowercase start after newline)
        normalized = re.sub(r'\n\s*([a-z])', r' \1', normalized)
        
        # Normalize multiple spaces to single space
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        
        # Normalize multiple newlines to double newline
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        
        # Fix question number spacing issues: "1 ." -> "1."
        normalized = re.sub(r'(\d+)\s+\.', r'\1.', normalized)
        normalized = re.sub(r'(\d+)\s+\)', r'\1)', normalized)
        
        # Fix option spacing: "( A )" -> "(A)"
        normalized = re.sub(r'\(\s*([A-Da-d])\s*\)', r'(\1)', normalized)
        
        # Normalize Q/Ques variations
        normalized = re.sub(r'Q\s*\.?\s*(\d+)', r'Q.\1', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'Ques\s*\.?\s*(\d+)', r'Ques.\1', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'Question\s+(\d+)', r'Question \1', normalized, flags=re.IGNORECASE)
        
        # Keep page markers to allow page-number inference for images
        
        # Remove common header/footer patterns
        normalized = re.sub(r'AFCAT\s*\d{4}', '', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'Air Force Common Admission Test', '', normalized, flags=re.IGNORECASE)
        
        return normalized.strip()
    
    def detect_paper_format(self, text: str) -> str:
        """
        Auto-detect the paper format based on text patterns.
        Returns: 'official', 'memory_based', 'scanned', or 'unknown'
        """
        text_sample = text[:5000]  # Analyze first part
        
        # Check for official paper markers
        if re.search(r'AFCAT.*Official|Booklet.*Series|Roll\s*Number', text_sample, re.IGNORECASE):
            return 'official'
        
        # Check for memory-based patterns (less structured)
        if re.search(r'Memory\s*Based|Recalled\s*Questions|As\s*per\s*Memory', text_sample, re.IGNORECASE):
            return 'memory_based'
        
        # Check OCR quality (scanned papers have more errors)
        error_indicators = len(re.findall(r'[|\\@#$%^&*]{2,}', text_sample))
        if error_indicators > 10:
            return 'scanned'
        
        # Detect based on question format
        if re.search(r'Q\.\d+|Question\s+\d+', text_sample):
            return 'memory_based'  # Memory-based often use Q.1 format
        elif re.search(r'^\s*\d+\.\s+[A-Z]', text_sample, re.MULTILINE):
            return 'official'  # Official uses clean "1. Question" format
        
        return 'unknown'
        
    def extract_questions(
        self,
        text: str,
        min_question_length: int = 10,
        use_fallback: bool = True
    ) -> List[ExtractedQuestion]:
        """
        Extract structured questions from OCR text.
        
        Args:
            text: Full text from OCR
            min_question_length: Minimum characters for valid question
            use_fallback: Use heuristic parser if regex yields few questions
            
        Returns:
            List of ExtractedQuestion objects
        """
        questions = []
        seen_numbers = set()
        
        # Normalize text first for better pattern matching
        normalized_text = self.normalize_text(text)

        # Page ranges for later page-number assignment (using markers from get_full_text)
        page_ranges = self._compute_page_ranges(normalized_text)

        # Pre-scan for figure-style questions and match-the-following blocks
        figure_candidates = self._detect_figure_candidates(normalized_text)
        match_following_questions = self._extract_match_following(normalized_text, seen_numbers)
        
        # Detect paper format for adaptive processing
        paper_format = self.detect_paper_format(text)
        logger.info(f"Detected paper format: {paper_format}")
        
        # Try each question pattern
        for pattern in self.compiled_q_patterns:
            matches = list(pattern.finditer(normalized_text))
            
            if matches:
                for i, match in enumerate(matches):
                    try:
                        q_num = int(match.group('num'))
                        q_text = match.group('text').strip()
                        
                        # Skip duplicates
                        if q_num in seen_numbers:
                            continue
                            
                        # Skip too short
                        if len(q_text) < min_question_length:
                            continue
                        
                        # Get the full text including options (until next question)
                        match_start = match.start()
                        if i + 1 < len(matches):
                            match_end = matches[i + 1].start()
                        else:
                            match_end = len(normalized_text)
                        
                        full_question_text = normalized_text[match_start:match_end]
                        
                        # Extract options from the full question area
                        options = self._extract_options(full_question_text)
                        
                        # Clean question text (remove option text)
                        clean_text = self._clean_question_text(q_text)

                        # Determine page number
                        page_num = self._locate_page(match_start, page_ranges)
                        
                        # Calculate confidence based on extraction quality
                        has_options = len(options) >= 2
                        has_good_text = len(clean_text.strip()) >= 30
                        extract_confidence = 0.5 + (0.25 if has_options else 0) + (0.25 if has_good_text else 0)
                        
                        if clean_text.strip():
                            questions.append(ExtractedQuestion(
                                question_number=q_num,
                                text=clean_text.strip(),
                                options=options,
                                page_num=page_num,
                                confidence=extract_confidence
                            ))
                            seen_numbers.add(q_num)
                            
                    except (ValueError, AttributeError):
                        continue
                        
                if len(questions) >= 20:  # Found enough with this pattern
                    break
        
        # Try Roman numeral pattern
        if len(questions) < 20:
            roman_matches = list(self.roman_pattern.finditer(normalized_text))
            for match in roman_matches:
                try:
                    roman_num = match.group('num').lower()
                    if roman_num in self.ROMAN_TO_INT:
                        q_num = self.ROMAN_TO_INT[roman_num]
                        if q_num not in seen_numbers:
                            q_text = match.group('text').strip()
                            if len(q_text) >= min_question_length:
                                options = self._extract_options(q_text)
                                clean_text = self._clean_question_text(q_text)
                                page_num = self._locate_page(match.start(), page_ranges)
                                # Confidence for Roman numeral match
                                has_options = len(options) >= 2
                                extract_confidence = 0.6 + (0.2 if has_options else 0)
                                questions.append(ExtractedQuestion(
                                    question_number=q_num,
                                    text=clean_text.strip(),
                                    options=options,
                                    page_num=page_num,
                                    confidence=extract_confidence
                                ))
                                seen_numbers.add(q_num)
                except (ValueError, AttributeError):
                    continue
        
        # RELAXED MATH PATTERNS: Try looser patterns for Q81-100 (Math zone)
        # These questions often fail due to broken numbering like "88 A train" (no dot)
        max_q_found = max(seen_numbers) if seen_numbers else 0
        if max_q_found >= 80:  # This is likely a full paper, try math patterns
            for pattern in self.compiled_math_patterns:
                matches = list(pattern.finditer(normalized_text))
                for match in matches:
                    try:
                        q_num = int(match.group('num'))
                        # Only apply to Math zone (81-100) and not already found
                        if 81 <= q_num <= 100 and q_num not in seen_numbers:
                            q_text = match.group('text').strip()
                            if len(q_text) >= min_question_length:
                                options = self._extract_options(q_text)
                                clean_text = self._clean_question_text(q_text)
                                page_num = self._locate_page(match.start(), page_ranges)
                                # Lower confidence for relaxed match
                                has_options = len(options) >= 2
                                extract_confidence = 0.45 + (0.2 if has_options else 0)
                                questions.append(ExtractedQuestion(
                                    question_number=q_num,
                                    text=clean_text.strip(),
                                    options=options,
                                    question_type="math_rescued",
                                    page_num=page_num,
                                    confidence=extract_confidence
                                ))
                                seen_numbers.add(q_num)
                                logger.debug(f"Relaxed Math pattern rescued Q{q_num}")
                    except (ValueError, AttributeError):
                        continue
        
        # Fallback: Line-by-line heuristic parsing
        if use_fallback and len(questions) < 20:
            logger.info(f"Using fallback parser (only {len(questions)} questions found)")
            fallback_questions = self._fallback_line_parser(normalized_text, seen_numbers, min_question_length)
            questions.extend(fallback_questions)

        # Add match-the-following questions detected separately
        if match_following_questions:
            questions.extend(match_following_questions)
            seen_numbers.update(q.question_number for q in match_following_questions)

        # Add figure placeholders not already captured
        for candidate in figure_candidates:
            if candidate['number'] not in seen_numbers:
                # Use confidence from detection method if available
                conf = candidate.get('confidence', 0.4)
                questions.append(ExtractedQuestion(
                    question_number=candidate['number'],
                    text=candidate['text'],
                    options=candidate.get('options', []),
                    question_type="non_verbal_figure",
                    has_diagram_reference=True,
                    confidence=conf
                ))
                seen_numbers.add(candidate['number'])

        # VACUUM SCRAPER: Rescue missing questions by grabbing text between known questions
        # This is especially useful for Math questions (Q81-100) with broken OCR
        rescued = self._vacuum_scrape_missing(normalized_text, questions, seen_numbers)
        if rescued:
            questions.extend(rescued)
            logger.info(f"Vacuum Scraper rescued {len(rescued)} additional questions")

        # Insert placeholders for missing internal numbers (likely figure/table questions)
        placeholders = self._add_missing_placeholders(questions, seen_numbers)
        if placeholders:
            questions.extend(placeholders)
        
        # FIX 2: Add trailing placeholders to reach exactly 100 questions for full papers
        max_q_found = max(seen_numbers) if seen_numbers else 0
        if max_q_found >= 80 and max_q_found < 100:
            trailing = self._add_trailing_placeholders(questions, seen_numbers, target=100)
            if trailing:
                questions.extend(trailing)
                logger.info(f"Added {len(trailing)} trailing placeholders to reach Q100")
        
        # Sort by question number
        questions.sort(key=lambda q: q.question_number)
        
        # OPTION B: Apply neighbor-based inference with confidence scoring
        questions = self._apply_neighbor_inference(questions)
        
        logger.info(f"Extracted {len(questions)} questions total")
        return questions
    
    def _apply_neighbor_inference(self, questions: List[ExtractedQuestion]) -> List[ExtractedQuestion]:
        """
        Option B Implementation: Infer context for placeholder/figure questions
        from their neighboring questions with robust confidence scoring.
        
        Confidence factors:
        - Same page as neighbors: +0.2
        - Neighbors have same section: +0.3
        - Consecutive question numbers: +0.15
        - Neighbors have similar topic keywords: +0.2
        - Both neighbors exist: +0.15
        """
        if len(questions) < 3:
            return questions
        
        # Section detection keywords for neighbor analysis
        section_keywords = {
            "verbal": ["synonym", "antonym", "idiom", "phrase", "comprehension", "passage", 
                       "sentence", "grammar", "word", "meaning", "vocabulary", "spelling",
                       "fill in", "blank", "cloze", "error", "correct"],
            "numerical": ["number", "percentage", "ratio", "profit", "loss", "average", 
                          "speed", "distance", "time", "interest", "age", "work", "pipe",
                          "train", "boat", "algebra", "geometry", "fraction", "decimal",
                          "lcm", "hcf", "simplify", "calculate"],
            "reasoning": ["analogy", "series", "pattern", "sequence", "missing", "odd one",
                          "direction", "blood relation", "coding", "decoding", "syllogism",
                          "statement", "conclusion", "assumption", "venn diagram", "cube",
                          "dice", "clock", "calendar", "mirror", "water", "paper folding",
                          "figure", "image", "embedded", "counting", "matrix"],
            "general_awareness": ["capital", "currency", "president", "minister", "award",
                                  "river", "mountain", "country", "state", "city", "war",
                                  "treaty", "constitution", "amendment", "scheme", "policy",
                                  "sports", "book", "author", "scientist", "invention",
                                  "discovery", "battle", "dynasty", "empire", "kingdom",
                                  "freedom", "independence", "movement"]
        }
        
        # Non-verbal reasoning specific keywords (figure questions)
        nonverbal_keywords = ["figure", "diagram", "pattern", "image", "mirror", "water",
                              "rotation", "paper folding", "embedded", "counting", "matrix",
                              "series", "analogy", "completion", "cube", "dice", "unfolded"]
        
        for idx, q in enumerate(questions):
            # Skip questions that don't need inference (already have good text)
            if q.question_type not in ("non_verbal_figure", "unknown") and q.confidence >= 0.6:
                continue
            if "[FIGURE" not in q.text and "[MISSING" not in q.text and len(q.text) > 50:
                continue
            
            # Find neighbors
            prev_q = questions[idx - 1] if idx > 0 else None
            next_q = questions[idx + 1] if idx < len(questions) - 1 else None
            
            confidence = 0.0
            inferred_section = "unknown"
            prev_topic_hint = ""
            next_topic_hint = ""
            
            # Factor 1: Both neighbors exist
            if prev_q and next_q:
                confidence += 0.15
            
            # Factor 2: Same page as neighbors
            pages = []
            if prev_q:
                pages.append(prev_q.page_num)
            if next_q:
                pages.append(next_q.page_num)
            if pages and q.page_num in pages:
                confidence += 0.2
            
            # Factor 3: Consecutive question numbers
            consecutive_count = 0
            if prev_q and prev_q.question_number == q.question_number - 1:
                consecutive_count += 1
            if next_q and next_q.question_number == q.question_number + 1:
                consecutive_count += 1
            confidence += consecutive_count * 0.075  # up to 0.15
            
            # Factor 4: Detect section from neighbor text
            neighbor_sections = []
            for neighbor in [prev_q, next_q]:
                if not neighbor:
                    continue
                text_lower = (neighbor.text + " " + " ".join(neighbor.options)).lower()
                for section, keywords in section_keywords.items():
                    for kw in keywords:
                        if kw in text_lower:
                            neighbor_sections.append(section)
                            if neighbor == prev_q:
                                prev_topic_hint = kw
                            else:
                                next_topic_hint = kw
                            break
            
            # If neighbors agree on section, high confidence
            if len(set(neighbor_sections)) == 1 and neighbor_sections:
                inferred_section = neighbor_sections[0]
                confidence += 0.3
            elif neighbor_sections:
                # Use most common section
                from collections import Counter
                inferred_section = Counter(neighbor_sections).most_common(1)[0][0]
                confidence += 0.15
            
            # Factor 5: Check for non-verbal reasoning context
            is_nonverbal = False
            for neighbor in [prev_q, next_q]:
                if not neighbor:
                    continue
                text_lower = (neighbor.text + " " + " ".join(neighbor.options)).lower()
                if any(kw in text_lower for kw in nonverbal_keywords):
                    is_nonverbal = True
                    confidence += 0.1
                    break
            
            # Apply inference if we have reasonable confidence
            if confidence >= 0.3:
                # Create neighbor context
                ctx = NeighborContext(
                    prev_question=prev_q,
                    next_question=next_q,
                    prev_topic_hint=prev_topic_hint,
                    next_topic_hint=next_topic_hint,
                    inferred_section=inferred_section,
                    confidence=min(confidence, 1.0)
                )
                q.neighbor_context = ctx
                
                # Update question type if it was a placeholder
                if q.question_type == "unknown" or "[FIGURE" in q.text or "[MISSING" in q.text:
                    if is_nonverbal or inferred_section == "reasoning":
                        q.question_type = "non_verbal_figure"
                        q.has_diagram_reference = True
                    elif inferred_section != "unknown":
                        q.question_type = f"inferred_{inferred_section}"
                
                # Update confidence
                q.confidence = max(q.confidence, confidence)
                
                logger.debug(
                    f"Q{q.question_number}: Inferred section={inferred_section}, "
                    f"confidence={confidence:.2f}, hints=[{prev_topic_hint}, {next_topic_hint}]"
                )
        
        return questions
    
    def _fallback_line_parser(
        self,
        text: str,
        seen_numbers: set,
        min_length: int = 10
    ) -> List[ExtractedQuestion]:
        """
        Heuristic line-by-line parser for difficult formats.
        Detects question starts and accumulates text until next question.
        """
        questions = []
        lines = text.split('\n')
        current_page = 0
        
        # Patterns to detect question starts
        q_start_patterns = [
            r'^\s*(\d{1,3})\s*[.):]',           # "1." "1)" "1:"
            r'^\s*Q\.?\s*(\d{1,3})',            # "Q1" "Q.1"
            r'^\s*\((\d{1,3})\)',               # "(1)"
            r'^\s*\[(\d{1,3})\]',               # "[1]"
            r'^\s*Question\s+(\d{1,3})',        # "Question 1"
            r'^\s*Ques\.?\s*(\d{1,3})',         # "Ques 1"
        ]
        
        current_question = None
        current_text = []
        
        for line in lines:
            # Page marker
            page_match = re.match(r'---\s*PAGE\s*(\d+)\s*---', line.strip(), re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1)) - 1  # zero-based
                continue

            line = line.strip()
            if not line:
                continue
            
            # Check if line starts a new question
            q_num = None
            for pattern in q_start_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    q_num = int(match.group(1))
                    break
            
            if q_num is not None and q_num not in seen_numbers:
                # Save previous question
                if current_question is not None and current_text:
                    full_text = ' '.join(current_text)
                    if len(full_text) >= min_length:
                        options = self._extract_options(full_text)
                        clean_text = self._clean_question_text(full_text)
                        # Fallback extraction confidence (lower than regex matches)
                        has_options = len(options) >= 2
                        has_good_text = len(clean_text.strip()) >= 30
                        extract_confidence = 0.4 + (0.2 if has_options else 0) + (0.15 if has_good_text else 0)
                        questions.append(ExtractedQuestion(
                            question_number=current_question,
                            text=clean_text.strip(),
                            options=options,
                            page_num=current_page,
                            confidence=extract_confidence
                        ))
                        seen_numbers.add(current_question)
                
                # Start new question
                current_question = q_num
                # Remove the question number prefix from line
                for pattern in q_start_patterns:
                    line = re.sub(pattern, '', line, flags=re.IGNORECASE).strip()
                current_text = [line] if line else []
            
            elif current_question is not None:
                # Continue accumulating text for current question
                current_text.append(line)
        
        # Don't forget the last question
        if current_question is not None and current_text:
            full_text = ' '.join(current_text)
            if len(full_text) >= min_length and current_question not in seen_numbers:
                options = self._extract_options(full_text)
                clean_text = self._clean_question_text(full_text)
                has_options = len(options) >= 2
                has_good_text = len(clean_text.strip()) >= 30
                extract_confidence = 0.4 + (0.2 if has_options else 0) + (0.15 if has_good_text else 0)
                questions.append(ExtractedQuestion(
                    question_number=current_question,
                    text=clean_text.strip(),
                    options=options,
                    page_num=current_page,
                    confidence=extract_confidence
                ))
        
        logger.info(f"Fallback parser found {len(questions)} additional questions")
        return questions
    
    def _vacuum_scrape_missing(
        self,
        full_text: str,
        questions: List[ExtractedQuestion],
        seen_numbers: set
    ) -> List[ExtractedQuestion]:
        """
        VACUUM SCRAPER: Rescue missing questions by grabbing text between known questions.
        
        If Q85 and Q87 are found, ALL text between them belongs to Q86.
        This is especially useful for Math questions (Q81-100) that fail regex parsing
        due to broken numbering or math symbols confusing OCR.
        
        For Math/GA zones (shuffled), we DON'T infer topic from neighbors.
        """
        if len(questions) < 2:
            return []
        
        rescued = []
        sorted_qs = sorted(questions, key=lambda q: q.question_number)
        
        # Map question numbers to their text positions in full_text
        q_positions = {}
        for q in sorted_qs:
            if not q.text or len(q.text) < 10:
                continue
            # Find where this question's text starts in the full document
            # Use first 50 chars to locate (avoid partial matches)
            search_text = q.text[:min(50, len(q.text))]
            pos = full_text.find(search_text)
            if pos != -1:
                q_positions[q.question_number] = {
                    'start': pos,
                    'end': pos + len(q.text),
                    'question': q
                }
        
        # Look for gaps between consecutive found questions
        q_nums = sorted(q_positions.keys())
        
        for i in range(len(q_nums) - 1):
            curr_num = q_nums[i]
            next_num = q_nums[i + 1]
            gap = next_num - curr_num
            
            # If there's a gap of 1-3 questions between found questions
            if 1 < gap <= 4:
                curr_q = q_positions[curr_num]
                next_q = q_positions[next_num]
                
                # Extract text between end of current Q and start of next Q
                between_start = curr_q['end']
                between_end = next_q['start']
                
                if between_end > between_start:
                    raw_text = full_text[between_start:between_end].strip()
                    
                    # Only rescue if there's meaningful content (>15 chars)
                    if len(raw_text) > 15:
                        # For each missing number in the gap
                        for missing_num in range(curr_num + 1, next_num):
                            if missing_num in seen_numbers:
                                continue
                            
                            # Determine zone and question type based on number
                            if 81 <= missing_num <= 100:
                                q_type = "math_inferred"
                                placeholder_text = f"[RESCUED MATH Q{missing_num}] {raw_text[:200]}"
                            elif 31 <= missing_num <= 55:
                                q_type = "ga_inferred"
                                placeholder_text = f"[RESCUED GA Q{missing_num}] {raw_text[:200]}"
                            elif 56 <= missing_num <= 80:
                                q_type = "non_verbal_figure"
                                placeholder_text = f"[RESCUED REASONING Q{missing_num}] {raw_text[:200]}"
                            else:  # 1-30
                                q_type = "verbal_inferred"
                                placeholder_text = f"[RESCUED VERBAL Q{missing_num}] {raw_text[:200]}"
                            
                            # Try to extract options from the raw text
                            options = self._extract_options(raw_text)
                            
                            # Create rescued question
                            rescued.append(ExtractedQuestion(
                                question_number=missing_num,
                                text=placeholder_text,
                                options=options,
                                question_type=q_type,
                                has_diagram_reference=(q_type == "non_verbal_figure"),
                                confidence=0.35,  # Low confidence - this is rescued content
                                page_num=curr_q['question'].page_num
                            ))
                            seen_numbers.add(missing_num)
                            
                            logger.debug(
                                f"🔍 Vacuum scraped Q{missing_num} between Q{curr_num} and Q{next_num} "
                                f"({len(raw_text)} chars)"
                            )
                            
                            # FIXED: Don't break - try to rescue ALL missing questions in the gap
                            # For multiple missing Qs, each gets the same raw_text (better than nothing)
                            # The classifier will assign proper zone-based topics
        
        return rescued

    def _extract_options(self, text: str) -> List[str]:
        """Extract options from question text."""
        for pattern in self.compiled_opt_patterns:
            matches = pattern.findall(text)
            if matches and len(matches) >= 2:
                # Sort by option letter
                sorted_matches = sorted(matches, key=lambda x: x[0].upper())
                return [opt[1].strip() for opt in sorted_matches]
        return []
    
    def _clean_question_text(self, text: str) -> str:
        """Remove options from question text to get clean question."""
        clean = text
        
        # Remove option patterns
        for pattern in self.compiled_opt_patterns:
            clean = pattern.sub('', clean)
            
        # Clean up extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        
        return clean.strip()

    def _detect_figure_candidates(self, text: str) -> List[Dict]:
        """
        Enhanced detection of figure/image questions using multiple strategies:
        1. Minimal text with option markers (number + A/B/C/D only)
        2. Keywords that indicate non-verbal reasoning content
        3. Short question text with diagram/figure references
        """
        candidates = []
        seen_nums = set()
        lines = text.split('\n')
        
        # Keywords indicating figure/diagram based questions
        figure_keywords = [
            # Pattern completion
            r"complete\s+the\s+(pattern|series|figure)",
            r"which\s+(figure|pattern)\s+(completes|comes\s+next)",
            r"find\s+the\s+missing\s+(figure|pattern|number)",
            # Mirror/water image
            r"mirror\s+image", r"water\s+image", r"reflection\s+of",
            # Paper folding/cutting
            r"paper\s+(folding|cutting|punching)", r"fold(ed)?\s+(paper|sheet)",
            # Cube and dice
            r"cube|dice|unfolded", r"opposite\s+(face|side)",
            # Counting figures
            r"count(ing)?\s+(the\s+)?(triangles?|squares?|figures?|lines?)",
            r"how\s+many\s+(triangles?|squares?|figures?)",
            # Embedded/hidden figures
            r"embedded\s+figure", r"hidden\s+(in|figure)",
            # Figure series/analogy
            r"figure\s+(series|analogy|matrix)",
            r"(choose|select|find)\s+the\s+(odd|different)\s+(figure|one)",
            # Rotation
            r"rotat(e|ed|ion)", r"position\s+of\s+the",
            # General references
            r"(see|refer|given|above|below)\s+(figure|diagram|image)",
            r"in\s+the\s+(given\s+)?(figure|diagram)",
            r"venn\s+diagram", r"shaded\s+(region|area|part)",
        ]
        
        # Strategy 1: Question number followed by only options (pure figure questions)
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            num_match = re.match(r'^\s*(?:Q\.?\s*)?(\d{1,3})\s*[.)]?(?:\s*[A-Da-d][.)]?)?\s*$', stripped)
            if num_match:
                q_num = int(num_match.group(1))
                if q_num in seen_nums:
                    continue

                following = lines[idx + 1: idx + 8]
                options = []
                for opt_line in following:
                    opt_line = opt_line.strip()
                    if not opt_line:
                        continue
                    if re.match(r'^\s*(?:Q\.?\s*)?\d{1,3}\s*[.)]', opt_line):
                        break
                    opt_match = re.match(r'^[\(\[]?([A-Da-d])[\)\].:-]?\s*(.*)$', opt_line)
                    if opt_match:
                        options.append(opt_match.group(2).strip())
                
                if len(options) >= 3:
                    candidates.append({
                        "number": q_num,
                        "options": options[:4],
                        "text": "[FIGURE QUESTION - non verbal reasoning pattern/diagram]",
                        "detection_method": "minimal_text",
                        "confidence": 0.5
                    })
                    seen_nums.add(q_num)
        
        # Strategy 2: Detect by non-verbal reasoning keywords
        for idx, line in enumerate(lines):
            stripped = line.strip()
            q_match = re.match(r'^\s*(?:Q\.?\s*)?(\d{1,3})\s*[.)]\s*(.+)$', stripped)
            if not q_match:
                continue
            
            q_num = int(q_match.group(1))
            q_text = q_match.group(2)
            
            if q_num in seen_nums:
                continue
            
            # Collect more text from following lines
            full_text = q_text
            for nxt in lines[idx + 1: idx + 4]:
                nxt = nxt.strip()
                if not nxt or re.match(r'^[\(\[]?[A-Da-d1-4][\)\].:-]', nxt):
                    break
                full_text += " " + nxt
            
            text_lower = full_text.lower()
            for kw_pattern in figure_keywords:
                if re.search(kw_pattern, text_lower):
                    options = []
                    for opt_line in lines[idx + 1: idx + 8]:
                        opt_line = opt_line.strip()
                        if not opt_line:
                            continue
                        opt_match = re.match(r'^[\(\[]?([A-Da-d])[\)\].:-]?\s*(.*)$', opt_line)
                        if opt_match:
                            options.append(opt_match.group(2).strip())
                    
                    candidates.append({
                        "number": q_num,
                        "options": options[:4] if options else [],
                        "text": f"{full_text.strip()} [FIGURE/DIAGRAM BASED]",
                        "detection_method": "keyword",
                        "confidence": 0.7
                    })
                    seen_nums.add(q_num)
                    break
        
        # Strategy 3: Very short question text (< 20 chars) without complete sentence
        for idx, line in enumerate(lines):
            stripped = line.strip()
            q_match = re.match(r'^\s*(?:Q\.?\s*)?(\d{1,3})\s*[.)]\s*(.{1,25})$', stripped)
            if not q_match:
                continue
            
            q_num = int(q_match.group(1))
            q_text = q_match.group(2).strip()
            
            if q_num in seen_nums:
                continue
            
            # Short text without question words likely means figure
            if len(q_text) < 15 and not re.search(r'\b(is|are|was|were|will|what|which|who|how)\b', q_text.lower()):
                options = []
                for opt_line in lines[idx + 1: idx + 8]:
                    opt_line = opt_line.strip()
                    if not opt_line:
                        continue
                    if re.match(r'^\s*(?:Q\.?\s*)?\d{1,3}\s*[.)]', opt_line):
                        break
                    opt_match = re.match(r'^[\(\[]?([A-Da-d])[\)\].:-]?\s*(.*)$', opt_line)
                    if opt_match:
                        options.append(opt_match.group(2).strip())
                
                if len(options) >= 2:
                    candidates.append({
                        "number": q_num,
                        "options": options[:4],
                        "text": f"{q_text} [LIKELY FIGURE - incomplete text]",
                        "detection_method": "short_text",
                        "confidence": 0.4
                    })
                    seen_nums.add(q_num)
        
        logger.info(f"Figure detection found {len(candidates)} candidates")
        return candidates

    def _extract_match_following(self, text: str, seen_numbers: set) -> List[ExtractedQuestion]:
        """Parse 'Match the Following' style blocks into structured questions."""
        results = []
        lines = text.split('\n')
        current_page = 0
        q_start_patterns = [
            r'^\s*(\d{1,3})\s*[.):]',
            r'^\s*Q\.?\s*(\d{1,3})',
            r'^\s*\((\d{1,3})\)',
            r'^\s*\[(\d{1,3})\]'
        ]

        i = 0
        while i < len(lines):
            line = lines[i]
            page_match = re.match(r'---\s*PAGE\s*(\d+)\s*---', line.strip(), re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1)) - 1
                i += 1
                continue
            if not re.search(r'match\s+the\s+following|match\s+list|list\s*i\b', line, re.IGNORECASE):
                i += 1
                continue

            # Look for question number on this line or previous line
            q_num = None
            search_window = [line]
            if i > 0:
                search_window.insert(0, lines[i - 1])
            for l in search_window:
                for pattern in q_start_patterns:
                    m = re.search(pattern, l, re.IGNORECASE)
                    if m:
                        q_num = int(m.group(1))
                        break
                if q_num:
                    break

            # If no explicit number, skip to avoid duplicates
            if q_num is None or q_num in seen_numbers:
                i += 1
                continue

            block_lines = [line.strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                # Stop if next question start appears
                if any(re.match(p, nxt.strip(), re.IGNORECASE) for p in q_start_patterns):
                    break
                block_lines.append(nxt.strip())
                j += 1

            block_text = '\n'.join([l for l in block_lines if l])
            options = self._extract_options(block_text)
            clean_text = self._clean_question_text(block_text)

            results.append(ExtractedQuestion(
                question_number=q_num,
                text=clean_text if clean_text else "Match the Following",
                options=options,
                question_type="match_following",
                has_diagram_reference=False,
                confidence=0.55,
                page_num=current_page
            ))
            seen_numbers.add(q_num)
            i = j

        return results

    def _add_missing_placeholders(self, questions: List[ExtractedQuestion], seen_numbers: set) -> List[ExtractedQuestion]:
        """
        Create placeholder entries for internal numbering gaps (likely figures).
        Uses neighbor context to infer page numbers and improve confidence.
        
        SAFE MODE: Sandwich Rule only applies to CLUSTERED zones (Verbal, Reasoning)
        and is DISABLED for SHUFFLED zones (GA, Math) to prevent misclassification.
        """
        if not questions:
            return []
        
        # Create lookup by question number for neighbor inference
        q_by_num = {q.question_number: q for q in questions}
        nums = sorted(q.question_number for q in questions)
        max_q = max(nums) if nums else 0
        is_full_paper = max_q >= 80  # Only apply zone logic for full papers
        
        placeholders = []
        
        for a, b in zip(nums, nums[1:]):
            if b - a <= 1 or b - a > 5:  # ignore tiny and huge gaps
                continue
            
            # Get neighbor questions for context
            prev_q = q_by_num.get(a)
            next_q = q_by_num.get(b)
            
            for missing in range(a + 1, b):
                if missing in seen_numbers:
                    continue
                
                # Infer page number from neighbors
                page_num = 0
                confidence = 0.2
                
                if prev_q and next_q:
                    # If both neighbors on same page, missing is likely there too
                    if prev_q.page_num == next_q.page_num:
                        page_num = prev_q.page_num
                        confidence = 0.4
                    else:
                        # Interpolate: if missing is closer to prev, use prev's page
                        if missing - a <= b - missing:
                            page_num = prev_q.page_num
                        else:
                            page_num = next_q.page_num
                        confidence = 0.3
                elif prev_q:
                    page_num = prev_q.page_num
                    confidence = 0.25
                elif next_q:
                    page_num = next_q.page_num
                    confidence = 0.25
                
                # SAFE MODE: Determine zone and apply conditional sandwich rule
                # Zone definitions: Q1-30=Verbal, Q31-55=GA, Q56-80=Reasoning, Q81-100=Math
                placeholder_text = "[MISSING / FIGURE QUESTION]"
                q_type = "inferred_gap"
                has_diagram = False
                
                if is_full_paper:
                    if 1 <= missing <= 30:
                        # VERBAL ZONE (Clustered) - Sandwich Rule ENABLED
                        # Passages, Cloze Tests are clustered in blocks
                        placeholder_text = "[VERBAL - likely passage/cloze question]"
                        q_type = "verbal_inferred"
                        has_diagram = False
                        confidence = max(confidence, 0.35)
                        
                    elif 31 <= missing <= 55:
                        # GA ZONE (Shuffled) - Sandwich Rule DISABLED
                        # DO NOT infer topic from neighbors - GK is random
                        placeholder_text = "[GA - topic cannot be inferred from neighbors]"
                        q_type = "ga_inferred"
                        has_diagram = False
                        confidence = 0.15  # Low confidence - shuffled zone
                        
                    elif 56 <= missing <= 80:
                        # REASONING ZONE (Clustered) - Sandwich Rule ENABLED
                        # Non-verbal figures come in blocks (5 Dot, 5 Embedded, etc.)
                        placeholder_text = "[REASONING - likely non-verbal figure/diagram]"
                        q_type = "non_verbal_figure"
                        has_diagram = True
                        confidence = max(confidence, 0.4)
                        
                    elif 81 <= missing <= 100:
                        # MATH ZONE (Shuffled) - Sandwich Rule DISABLED
                        # Math topics are random (Trains next to Profit next to Algebra)
                        placeholder_text = "[MATH - topic cannot be inferred from neighbors]"
                        q_type = "math_inferred"
                        has_diagram = False
                        confidence = 0.15  # Low confidence - shuffled zone
                else:
                    # Not a full paper - use question number to infer zone anyway
                    # This prevents defaulting everything to Reasoning
                    if 1 <= missing <= 30:
                        placeholder_text = "[VERBAL - inferred from position]"
                        q_type = "verbal_inferred"
                        has_diagram = False
                    elif 31 <= missing <= 55:
                        placeholder_text = "[GA - inferred from position]"
                        q_type = "ga_inferred"
                        has_diagram = False
                    elif 56 <= missing <= 80:
                        placeholder_text = "[REASONING - inferred from position]"
                        q_type = "non_verbal_figure"
                        has_diagram = True
                    elif 81 <= missing <= 100:
                        placeholder_text = "[MATH - inferred from position]"
                        q_type = "math_inferred"
                        has_diagram = False
                    else:
                        placeholder_text = "[UNKNOWN - position out of range]"
                        q_type = "unknown_gap"
                        has_diagram = False
                
                placeholders.append(ExtractedQuestion(
                    question_number=missing,
                    text=placeholder_text,
                    options=[],
                    question_type=q_type,
                    has_diagram_reference=has_diagram,
                    confidence=confidence,
                    page_num=page_num
                ))
                seen_numbers.add(missing)
        
        if placeholders:
            logger.info(f"Added {len(placeholders)} placeholder questions for gaps (Safe Mode enabled)")
        
        return placeholders

    def _add_trailing_placeholders(self, questions: List[ExtractedQuestion], seen_numbers: set, target: int = 100) -> List[ExtractedQuestion]:
        """
        FIX 2: Add trailing placeholders for questions beyond max_found up to target.
        This ensures we always reach exactly 100 questions for full papers.
        
        Uses zone-based type assignment just like internal placeholders.
        """
        if not questions:
            return []
        
        max_q_found = max(q.question_number for q in questions)
        if max_q_found >= target:
            return []  # Already have enough
        
        trailing = []
        
        for num in range(max_q_found + 1, target + 1):
            if num in seen_numbers:
                continue
            
            # Zone-based type assignment (AFCAT section zones)
            # Q1-30=Verbal, Q31-55=GA, Q56-80=Reasoning, Q81-100=Math
            if 1 <= num <= 30:
                placeholder_text = "[VERBAL - trailing placeholder]"
                q_type = "verbal_inferred"
                has_diagram = False
            elif 31 <= num <= 55:
                placeholder_text = "[GA - trailing placeholder]"
                q_type = "ga_inferred"
                has_diagram = False
            elif 56 <= num <= 80:
                placeholder_text = "[REASONING - trailing placeholder]"
                q_type = "non_verbal_figure"
                has_diagram = True
            else:  # 81-100
                placeholder_text = "[MATH - trailing placeholder]"
                q_type = "math_inferred"
                has_diagram = False
            
            trailing.append(ExtractedQuestion(
                question_number=num,
                text=placeholder_text,
                options=[],
                question_type=q_type,
                has_diagram_reference=has_diagram,
                confidence=0.1,  # Low confidence for trailing placeholders
                page_num=0  # Unknown page
            ))
            seen_numbers.add(num)
        
        if trailing:
            logger.info(f"Added {len(trailing)} trailing placeholders (Q{max_q_found + 1}-Q{target})")
        
        return trailing

    def _compute_page_ranges(self, text: str) -> List[Tuple[int, int, int]]:
        """Return list of (start_idx, end_idx, page_num) inferred from markers."""
        ranges = []
        markers = list(re.finditer(r'---\s*PAGE\s*(\d+)\s*---', text, re.IGNORECASE))
        if not markers:
            ranges.append((0, len(text), 0))
            return ranges
        for idx, m in enumerate(markers):
            start = m.end()
            end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
            page_num = int(m.group(1)) - 1  # zero-based
            ranges.append((start, end, page_num))
        return ranges

    def _locate_page(self, position: int, ranges: List[Tuple[int, int, int]]) -> int:
        """Find page num for a character position based on page ranges."""
        for start, end, p in ranges:
            if start <= position < end:
                return p
        return 0
    
    def extract_with_context(
        self,
        text: str,
        include_surrounding: bool = True
    ) -> List[Dict]:
        """
        Extract questions with additional context.
        
        Returns:
            List of dicts with question, options, and context
        """
        questions = self.extract_questions(text)
        results = []
        
        lines = text.split('\n')
        
        for q in questions:
            result = {
                'number': q.question_number,
                'text': q.text,
                'options': q.options,
                'has_options': len(q.options) >= 2,
                'text_length': len(q.text),
                'word_count': len(q.text.split()),
            }
            
            # Detect if question has mathematical content
            result['has_math'] = bool(re.search(r'[\d+\-×÷*/=<>%°²³]', q.text))
            
            # Detect if question references a passage/figure
            result['has_reference'] = bool(
                re.search(r'passage|figure|diagram|table|graph|above|below', q.text, re.I)
            )
            
            results.append(result)
            
        return results


# Convenience functions
def quick_extract_pdf(pdf_path: str, engine: str = "easyocr") -> List[ExtractedQuestion]:
    """
    Quick extraction from PDF file.
    
    Example:
        questions = quick_extract_pdf("afcat_2024.pdf")
        for q in questions:
            print(f"Q{q.question_number}: {q.text[:50]}...")
    """
    ocr_engine = OCREngine(engine)
    ocr = ExamPaperOCR(engine=ocr_engine)
    extractor = MCQExtractor()
    
    # Extract text
    results = ocr.extract_from_pdf(pdf_path)
    full_text = ocr.get_full_text(results)
    
    # Extract questions
    questions = extractor.extract_questions(full_text)
    
    return questions


def quick_extract_image(image_path: str, engine: str = "easyocr") -> List[ExtractedQuestion]:
    """
    Quick extraction from image file.
    
    Example:
        questions = quick_extract_image("afcat_page1.jpg")
    """
    ocr_engine = OCREngine(engine)
    ocr = ExamPaperOCR(engine=ocr_engine)
    extractor = MCQExtractor()
    
    results = ocr.extract_from_image(image_path)
    full_text = '\n'.join(r.text for r in results)
    
    questions = extractor.extract_questions(full_text)
    
    return questions
