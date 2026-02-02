"""
Unit tests for plate text normalization.

Tests the PlateTextNormalizer domain service.
"""

import pytest

from app.domain.services import PlateTextNormalizer


class TestPlateTextNormalizer:
    """Tests for PlateTextNormalizer."""
    
    @pytest.fixture
    def normalizer(self) -> PlateTextNormalizer:
        """Create normalizer instance."""
        return PlateTextNormalizer()
    
    def test_normalize_basic(self, normalizer: PlateTextNormalizer):
        """Test basic normalization."""
        assert normalizer.normalize("MH12AB1234") == "MH12AB1234"
    
    def test_normalize_lowercase(self, normalizer: PlateTextNormalizer):
        """Test lowercase to uppercase conversion."""
        assert normalizer.normalize("mh12ab1234") == "MH12AB1234"
    
    def test_normalize_spaces(self, normalizer: PlateTextNormalizer):
        """Test space removal."""
        assert normalizer.normalize("MH 12 AB 1234") == "MH12AB1234"
    
    def test_normalize_special_chars(self, normalizer: PlateTextNormalizer):
        """Test special character removal."""
        assert normalizer.normalize("MH-12-AB-1234") == "MH12AB1234"
        assert normalizer.normalize("MH.12.AB.1234") == "MH12AB1234"
    
    def test_normalize_ocr_substitution_o_to_0(self, normalizer: PlateTextNormalizer):
        """Test O to 0 substitution in digit positions."""
        # O in last 4 positions (should be digits) → 0
        result = normalizer.normalize("MH12ABO234")
        assert result == "MH12AB0234"
    
    def test_normalize_ocr_substitution_i_to_1(self, normalizer: PlateTextNormalizer):
        """Test I to 1 substitution in digit positions."""
        result = normalizer.normalize("MH12ABI234")
        assert result == "MH12AB1234"
    
    def test_normalize_ocr_substitution_digit_to_letter(self, normalizer: PlateTextNormalizer):
        """Test digit to letter substitution in letter positions."""
        # 0 in first 2 positions (should be letters) → O
        result = normalizer.normalize("0H12AB1234")
        assert result == "OH12AB1234"
    
    def test_normalize_empty_string(self, normalizer: PlateTextNormalizer):
        """Test empty string handling."""
        assert normalizer.normalize("") == ""
    
    def test_normalize_short_string(self, normalizer: PlateTextNormalizer):
        """Test short string handling (less than 4 chars)."""
        assert normalizer.normalize("MH") == "MH"
        assert normalizer.normalize("MH1") == "MH1"
    
    def test_normalize_mixed_case(self, normalizer: PlateTextNormalizer):
        """Test mixed case with spaces."""
        result = normalizer.normalize("Mh 12 aB 1234")
        assert result == "MH12AB1234"
    
    def test_normalize_bharat_series(self, normalizer: PlateTextNormalizer):
        """Test Bharat series plate normalization."""
        result = normalizer.normalize("BH 01 AA 1234")
        assert result == "BH01AA1234"
