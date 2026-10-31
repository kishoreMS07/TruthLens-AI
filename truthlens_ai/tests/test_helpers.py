import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.main import _map_prediction, _safe_label

def test_ai_label():
    result = _map_prediction([{"label": "AI-generated", "score": 0.91}, {"label": "human", "score": 0.09}])
    assert result["status"] == "ai_indicated"
    assert result["score"] == 0.91

def test_human_label():
    result = _map_prediction([{"label": "human", "score": 0.82}, {"label": "AI-generated", "score": 0.18}])
    assert result["status"] == "human_indicated"

def test_unknown_labels_are_inconclusive():
    result = _map_prediction([{"label": "class 0", "score": 0.8}])
    assert result["status"] == "inconclusive"
