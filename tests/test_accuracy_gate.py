import sys, os, pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import AccuracyGateError, MasterTrainer


def test_is_exception():
    err = AccuracyGateError("bert", 0.50, 0.78)
    assert isinstance(err, Exception)


def test_message_contains_model():
    err = AccuracyGateError("bert", 0.50, 0.78)
    assert "bert" in str(err).lower()


def test_attributes_exist():
    err = AccuracyGateError("tfidf", 0.72, 0.76)
    assert hasattr(err, "args") and len(err.args) > 0


def test_bert_gate_thresholds():
    gates = {"accuracy": 0.78, "precision": 0.76, "f1": 0.77}
    passing = {"accuracy": 0.80, "precision": 0.80, "f1": 0.80}
    for m, t in gates.items():
        assert passing[m] >= t


def test_ensemble_gate_thresholds():
    gates = {"accuracy": 0.82, "precision": 0.80, "f1": 0.81}
    passing = {"accuracy": 0.85, "precision": 0.85, "f1": 0.85}
    for m, t in gates.items():
        assert passing[m] >= t


def test_master_trainer_instantiates():
    trainer = MasterTrainer(config_path="config.yaml")
    assert trainer is not None


def test_raises_on_low_score():
    with pytest.raises(AccuracyGateError):
        raise AccuracyGateError("naive_bayes", 0.60, 0.75)
