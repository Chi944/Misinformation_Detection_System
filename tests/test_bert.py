import sys, os, pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.bert_classifier import BERTMisinformationClassifier


@pytest.fixture
def model():
    return BERTMisinformationClassifier(dropout=0.3)


def test_instantiation(model):
    assert model is not None


def test_is_nn_module(model):
    import torch.nn as nn

    assert isinstance(model, nn.Module)


def test_has_classifier(model):
    # The wrapped model exposes the classifier head as model.classifier
    assert hasattr(model.model, "classifier")


def test_has_bert_encoder(model):
    # The underlying encoder is accessible as model.bert on the HF model
    assert hasattr(model.model, "bert")


def test_forward_shape():
    import torch

    m = BERTMisinformationClassifier(dropout=0.0)
    ids = torch.zeros(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    assert out.shape == (2, 2), "expected (2,2) got %s" % str(out.shape)


def test_forward_finite():
    import torch

    m = BERTMisinformationClassifier(dropout=0.0)
    ids = torch.zeros(1, 8, dtype=torch.long)
    mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    assert torch.isfinite(out).all()


def test_has_dropout(model):
    assert (
        hasattr(model, "dropout")
        or hasattr(model, "drop")
        or any("dropout" in n for n, _ in model.named_modules())
    )
