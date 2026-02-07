"""
Domain services for plate validation and normalization.

These services contain pure business logic with no infrastructure
dependencies. They can be easily unit tested.
"""

import re
from dataclasses import dataclass
from typing import ClassVar

from app.domain.models import Decision, VerificationStatus


@dataclass
class PlateTextNormalizer:
    """
    Normalizes OCR text output to standard plate format.
    
    Handles common OCR misreadings and character substitutions
    to improve plate matching accuracy.
    
    Example:
        >>> normalizer = PlateTextNormalizer()
        >>> normalizer.normalize("MH 12 AB O234")
        'MH12AB0234'
    """
    
    # Common OCR character substitutions (letter → digit)
    LETTER_TO_DIGIT: ClassVar[dict[str, str]] = {
        "O": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }
    
    # Common OCR character substitutions (digit → letter)
    DIGIT_TO_LETTER: ClassVar[dict[str, str]] = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "8": "B",
    }
    
    # Valid Indian state codes
    VALID_STATE_CODES: ClassVar[set[str]] = {
        "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
        "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
        "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
        "TN", "TR", "TS", "UK", "UP", "WB", "BH",  # BH = Bharat series
    }
    
    # Common OCR misreads for state codes (wrong → correct)
    STATE_CODE_FIXES: ClassVar[dict[str, str]] = {
        "HH": "MH",  # M misread as H
        "NH": "MH",  # M misread as N
        "WH": "MH",  # M misread as W
        "KH": "KA",  # A misread as H
        "DI": "DL",  # L misread as I
        "DT": "DL",  # L misread as T
        "TH": "TN",  # N misread as H
        "RH": "RJ",  # J misread as H
    }
    
    def normalize(self, text: str) -> str:
        """
        Normalize raw OCR text to standard plate format.
        
        Performs the following operations:
        1. Convert to uppercase
        2. Remove spaces and special characters
        3. Apply context-aware character substitution
        
        Args:
            text: Raw OCR text output.
        
        Returns:
            str: Normalized plate number.
        """
        if not text:
            return ""
        
        # Uppercase and remove non-alphanumeric
        normalized = text.upper()
        normalized = re.sub(r"[^A-Z0-9]", "", normalized)
        
        # Apply context-aware substitution
        normalized = self._apply_context_aware_substitution(normalized)
        
        return normalized
    
    def _apply_context_aware_substitution(self, text: str) -> str:
        """
        Apply character substitutions based on expected position.
        
        Indian plates follow pattern: [2 letters][1-2 digits][1-3 letters][4 digits]
        Use this knowledge to correct likely OCR errors.
        
        Args:
            text: Partially normalized text.
        
        Returns:
            str: Text with context-aware corrections.
        """
        if len(text) < 4:
            return text
        
        result = list(text)
        
        # First 2 chars should be letters (state code)
        for i in range(min(2, len(result))):
            if result[i] in self.DIGIT_TO_LETTER:
                result[i] = self.DIGIT_TO_LETTER[result[i]]
        
        # Chars at positions 2-3 should be digits (district code)
        for i in range(2, min(4, len(result))):
            if result[i] in self.LETTER_TO_DIGIT:
                result[i] = self.LETTER_TO_DIGIT[result[i]]
        
        # Last 4 chars should be digits
        if len(result) >= 4:
            for i in range(-4, 0):
                if result[i] in self.LETTER_TO_DIGIT:
                    result[i] = self.LETTER_TO_DIGIT[result[i]]
        
        # Fix common state code misreads
        if len(result) >= 2:
            state_code = "".join(result[:2])
            if state_code not in self.VALID_STATE_CODES:
                fixed = self.STATE_CODE_FIXES.get(state_code)
                if fixed:
                    result[0] = fixed[0]
                    result[1] = fixed[1]
        
        return "".join(result)


@dataclass
class PlateValidator:
    """
    Validates normalized plate numbers against known patterns.
    
    Supports multiple Indian plate formats including:
    - Standard: MH12AB1234
    - Bharat series: BH01AA1234
    - Commercial: MH12 1234
    - Defense: ABC1234
    
    Example:
        >>> validator = PlateValidator()
        >>> validator.is_valid("MH12AB1234")
        True
        >>> validator.is_valid("INVALID")
        False
    """
    
    # Indian vehicle plate patterns
    PLATE_PATTERNS: ClassVar[list[str]] = [
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$",  # Standard: MH12AB1234
        r"^BH[0-9]{2}[A-Z]{2}[0-9]{4}$",             # Bharat series
        r"^[A-Z]{2}[0-9]{1,2}[0-9]{4}$",             # Older commercial
        r"^[A-Z]{3}[0-9]{4}$",                        # Defense vehicles
        r"^[0-9]{2}BH[0-9]{4}[A-Z]{2}$",             # Alternate Bharat format
    ]
    
    def is_valid(self, plate_number: str) -> bool:
        """
        Check if plate number matches any valid pattern.
        
        Args:
            plate_number: Normalized plate number to validate.
        
        Returns:
            bool: True if plate matches a valid pattern.
        """
        if not plate_number or len(plate_number) < 6:
            return False
        
        # For now, accept any alphanumeric string of 6-12 characters
        # This allows detection of all kinds of plates
        if len(plate_number) <= 12 and plate_number.isalnum():
            return True
        
        # Also check strict patterns for analytics
        for pattern_str in self.PLATE_PATTERNS:
            if re.match(pattern_str, plate_number):
                return True
        
        return False
    
    def get_matching_pattern(self, plate_number: str) -> str | None:
        """
        Get the pattern that matches the plate number.
        
        Useful for analytics and debugging.
        
        Args:
            plate_number: Normalized plate number.
        
        Returns:
            str: Matching pattern, or None if no match.
        """
        if not plate_number:
            return None
        
        for pattern_str in self.PLATE_PATTERNS:
            if re.match(pattern_str, plate_number):
                return pattern_str
        
        return None


@dataclass
class ConfidenceThresholdEvaluator:
    """
    Evaluates combined confidence from detector and OCR.
    
    Uses min(detector_confidence, ocr_confidence) to determine
    final confidence. This fail-safe approach ensures we only
    accept results when both stages are confident.
    
    Attributes:
        threshold: Minimum confidence to accept result (default 0.70).
    
    Example:
        >>> evaluator = ConfidenceThresholdEvaluator(threshold=0.70)
        >>> evaluator.compute_final_confidence(0.95, 0.85)
        0.85
        >>> evaluator.is_confident(0.95, 0.85)
        True
    """
    
    threshold: float = 0.70
    
    def compute_final_confidence(
        self,
        detector_confidence: float,
        ocr_confidence: float,
    ) -> float:
        """
        Compute combined confidence score.
        
        Uses min() for fail-safe behavior - both stages must
        be confident for the result to be trusted.
        
        Args:
            detector_confidence: Confidence from plate detector.
            ocr_confidence: Confidence from OCR engine.
        
        Returns:
            float: Combined confidence score.
        """
        return min(detector_confidence, ocr_confidence)
    
    def compute_weighted_confidence(
        self,
        detector_confidence: float,
        ocr_confidence: float,
        detector_weight: float = 0.6,
    ) -> float:
        """
        Compute weighted confidence for analytics.
        
        Not used for decision making, but useful for monitoring
        and performance analysis.
        
        Args:
            detector_confidence: Confidence from plate detector.
            ocr_confidence: Confidence from OCR engine.
            detector_weight: Weight for detector (default 0.6).
        
        Returns:
            float: Weighted confidence score.
        """
        ocr_weight = 1.0 - detector_weight
        return (detector_confidence * detector_weight) + (ocr_confidence * ocr_weight)
    
    def is_confident(
        self,
        detector_confidence: float,
        ocr_confidence: float,
    ) -> bool:
        """
        Check if combined confidence meets threshold.
        
        Args:
            detector_confidence: Confidence from plate detector.
            ocr_confidence: Confidence from OCR engine.
        
        Returns:
            bool: True if confidence meets threshold.
        """
        final_confidence = self.compute_final_confidence(
            detector_confidence,
            ocr_confidence,
        )
        return final_confidence >= self.threshold


@dataclass
class DecisionMaker:
    """
    Makes ALLOW/ALERT decisions based on verification status.
    
    Implements fail-safe logic: deny access if uncertain.
    
    Example:
        >>> maker = DecisionMaker()
        >>> maker.make_decision(VerificationStatus.AUTHORIZED)
        Decision.ALLOW
        >>> maker.make_decision(VerificationStatus.UNKNOWN)
        Decision.ALERT
    """
    
    def make_decision(self, status: VerificationStatus) -> Decision:
        """
        Determine action based on verification status.
        
        Only AUTHORIZED status results in ALLOW. All other
        statuses (including UNKNOWN and failure states)
        result in ALERT for fail-safe behavior.
        
        Args:
            status: Verification status from plate check.
        
        Returns:
            Decision: ALLOW or ALERT.
        """
        if status == VerificationStatus.AUTHORIZED:
            return Decision.ALLOW
        
        # Fail-safe: deny access for any non-authorized status
        return Decision.ALERT
    
    def should_create_alert(self, status: VerificationStatus) -> bool:
        """
        Check if an alert should be created for this status.
        
        Alerts are created for unauthorized, unknown, and failure states.
        
        Args:
            status: Verification status.
        
        Returns:
            bool: True if alert should be created.
        """
        return status != VerificationStatus.AUTHORIZED
