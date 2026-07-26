from pathlib import Path

import pytest

from fraud_detection.inference import FraudPredictor


def test_missing_artifacts_raise_error(tmp_path: Path):
    with pytest.raises(Exception):
        FraudPredictor(tmp_path)
