"""
Unit tests for decision logic.

Tests the DecisionMaker and ConfidenceThresholdEvaluator domain services.
"""

import pytest

from app.domain.models import Decision, VerificationStatus
from app.domain.services import ConfidenceThresholdEvaluator, DecisionMaker


class TestDecisionMaker:
    """Tests for DecisionMaker."""
    
    @pytest.fixture
    def decision_maker(self) -> DecisionMaker:
        """Create decision maker instance."""
        return DecisionMaker()
    
    def test_authorized_allows(self, decision_maker: DecisionMaker):
        """Test AUTHORIZED status results in ALLOW."""
        decision = decision_maker.make_decision(VerificationStatus.AUTHORIZED)
        assert decision == Decision.ALLOW
    
    def test_unauthorized_alerts(self, decision_maker: DecisionMaker):
        """Test UNAUTHORIZED status results in ALERT."""
        decision = decision_maker.make_decision(VerificationStatus.UNAUTHORIZED)
        assert decision == Decision.ALERT
    
    def test_unknown_alerts(self, decision_maker: DecisionMaker):
        """Test UNKNOWN status results in ALERT (fail-safe)."""
        decision = decision_maker.make_decision(VerificationStatus.UNKNOWN)
        assert decision == Decision.ALERT
    
    def test_ocr_failed_alerts(self, decision_maker: DecisionMaker):
        """Test OCR_FAILED status results in ALERT."""
        decision = decision_maker.make_decision(VerificationStatus.OCR_FAILED)
        assert decision == Decision.ALERT
    
    def test_detection_failed_alerts(self, decision_maker: DecisionMaker):
        """Test DETECTION_FAILED status results in ALERT."""
        decision = decision_maker.make_decision(VerificationStatus.DETECTION_FAILED)
        assert decision == Decision.ALERT
    
    def test_should_create_alert_authorized(self, decision_maker: DecisionMaker):
        """Test no alert for authorized."""
        assert decision_maker.should_create_alert(VerificationStatus.AUTHORIZED) is False
    
    def test_should_create_alert_unauthorized(self, decision_maker: DecisionMaker):
        """Test alert created for unauthorized."""
        assert decision_maker.should_create_alert(VerificationStatus.UNAUTHORIZED) is True
    
    def test_should_create_alert_unknown(self, decision_maker: DecisionMaker):
        """Test alert created for unknown."""
        assert decision_maker.should_create_alert(VerificationStatus.UNKNOWN) is True


class TestConfidenceThresholdEvaluator:
    """Tests for ConfidenceThresholdEvaluator."""
    
    @pytest.fixture
    def evaluator(self) -> ConfidenceThresholdEvaluator:
        """Create evaluator with default threshold."""
        return ConfidenceThresholdEvaluator(threshold=0.70)
    
    def test_compute_final_confidence_min(self, evaluator: ConfidenceThresholdEvaluator):
        """Test final confidence is min of detector and OCR."""
        result = evaluator.compute_final_confidence(0.95, 0.85)
        assert result == 0.85
        
        result = evaluator.compute_final_confidence(0.80, 0.90)
        assert result == 0.80
    
    def test_compute_final_confidence_equal(self, evaluator: ConfidenceThresholdEvaluator):
        """Test when both confidences are equal."""
        result = evaluator.compute_final_confidence(0.85, 0.85)
        assert result == 0.85
    
    def test_is_confident_above_threshold(self, evaluator: ConfidenceThresholdEvaluator):
        """Test confident when both above threshold."""
        assert evaluator.is_confident(0.90, 0.85) is True
        assert evaluator.is_confident(0.75, 0.80) is True
    
    def test_is_confident_below_threshold(self, evaluator: ConfidenceThresholdEvaluator):
        """Test not confident when min below threshold."""
        assert evaluator.is_confident(0.90, 0.60) is False
        assert evaluator.is_confident(0.50, 0.90) is False
    
    def test_is_confident_at_threshold(self, evaluator: ConfidenceThresholdEvaluator):
        """Test confident at exact threshold."""
        assert evaluator.is_confident(0.70, 0.80) is True
        assert evaluator.is_confident(0.80, 0.70) is True
    
    def test_is_confident_just_below_threshold(self, evaluator: ConfidenceThresholdEvaluator):
        """Test not confident just below threshold."""
        assert evaluator.is_confident(0.69, 0.90) is False
    
    def test_compute_weighted_confidence(self, evaluator: ConfidenceThresholdEvaluator):
        """Test weighted confidence calculation."""
        # Default weights: 0.6 detector, 0.4 OCR
        result = evaluator.compute_weighted_confidence(1.0, 0.0)
        assert result == pytest.approx(0.6)
        
        result = evaluator.compute_weighted_confidence(0.0, 1.0)
        assert result == pytest.approx(0.4)
        
        result = evaluator.compute_weighted_confidence(0.90, 0.80)
        expected = (0.90 * 0.6) + (0.80 * 0.4)
        assert result == pytest.approx(expected)
    
    def test_custom_threshold(self):
        """Test evaluator with custom threshold."""
        strict_evaluator = ConfidenceThresholdEvaluator(threshold=0.90)
        assert strict_evaluator.is_confident(0.95, 0.92) is True
        assert strict_evaluator.is_confident(0.95, 0.85) is False
        
        lenient_evaluator = ConfidenceThresholdEvaluator(threshold=0.50)
        assert lenient_evaluator.is_confident(0.55, 0.60) is True
