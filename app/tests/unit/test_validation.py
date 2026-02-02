"""
Unit tests for plate validation.

Tests the PlateValidator domain service.
"""

import pytest

from app.domain.services import PlateValidator


class TestPlateValidator:
    """Tests for PlateValidator."""
    
    @pytest.fixture
    def validator(self) -> PlateValidator:
        """Create validator instance."""
        return PlateValidator()
    
    # Standard format tests: MH12AB1234
    
    def test_valid_standard_format(self, validator: PlateValidator):
        """Test standard Indian plate format."""
        assert validator.is_valid("MH12AB1234") is True
        assert validator.is_valid("DL01CA9999") is True
        assert validator.is_valid("KA05MN0001") is True
    
    def test_valid_single_digit_district(self, validator: PlateValidator):
        """Test single digit district code."""
        assert validator.is_valid("MH1AB1234") is True
    
    def test_valid_single_letter_series(self, validator: PlateValidator):
        """Test single letter series."""
        assert validator.is_valid("MH12A1234") is True
    
    def test_valid_three_letter_series(self, validator: PlateValidator):
        """Test three letter series."""
        assert validator.is_valid("MH12ABC1234") is True
    
    # Bharat series tests: BH01AA1234
    
    def test_valid_bharat_series(self, validator: PlateValidator):
        """Test Bharat series format."""
        assert validator.is_valid("BH01AA1234") is True
        assert validator.is_valid("BH99ZZ9999") is True
    
    # Commercial plate tests: MH12 1234
    
    def test_valid_commercial_format(self, validator: PlateValidator):
        """Test older commercial plate format."""
        assert validator.is_valid("MH121234") is True
        assert validator.is_valid("DL11234") is True
    
    # Defense vehicle tests: ABC1234
    
    def test_valid_defense_format(self, validator: PlateValidator):
        """Test defense vehicle plate format."""
        assert validator.is_valid("ABC1234") is True
        assert validator.is_valid("XYZ9999") is True
    
    # Invalid format tests
    
    def test_invalid_empty(self, validator: PlateValidator):
        """Test empty string is invalid."""
        assert validator.is_valid("") is False
    
    def test_invalid_too_short(self, validator: PlateValidator):
        """Test too short plate is invalid."""
        assert validator.is_valid("MH12") is False
        assert validator.is_valid("AB123") is False
    
    def test_invalid_random_text(self, validator: PlateValidator):
        """Test random text is invalid."""
        assert validator.is_valid("HELLO") is False
        assert validator.is_valid("12345678") is False
    
    def test_invalid_wrong_pattern(self, validator: PlateValidator):
        """Test wrong pattern combinations."""
        assert validator.is_valid("1234AB5678") is False
        assert validator.is_valid("ABCD1234EFGH") is False
    
    # Pattern matching tests
    
    def test_get_matching_pattern_standard(self, validator: PlateValidator):
        """Test correct pattern is identified for standard plates."""
        pattern = validator.get_matching_pattern("MH12AB1234")
        assert pattern is not None
        assert "A-Z" in pattern
    
    def test_get_matching_pattern_bharat(self, validator: PlateValidator):
        """Test correct pattern is identified for Bharat series."""
        pattern = validator.get_matching_pattern("BH01AA1234")
        assert pattern is not None
        assert "BH" in pattern
    
    def test_get_matching_pattern_none(self, validator: PlateValidator):
        """Test None returned for invalid plate."""
        assert validator.get_matching_pattern("INVALID") is None
        assert validator.get_matching_pattern("") is None
