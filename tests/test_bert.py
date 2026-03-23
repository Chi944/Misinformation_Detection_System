import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.bert_classifier import BERTClassifier  # noqa: E402


@pytest.fixture
def model():
    return BERTClassifier(config={"train_fresh": True})


def test_instantiation(model):
    assert model is not None


def test_is_nn_module(model):
    import torch.nn as nn

    assert isinstance(model, nn.Module)


def test_has_classifier(model):
    assert hasattr(model, "classifier")


def test_has_bert_encoder(model):
    assert hasattr(model, "bert")


def test_forward_shape():
    import torch

    m = BERTClassifier(config={"train_fresh": True})
    ids = torch.zeros(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    assert out.shape == (2, 2), "expected (2,2) got %s" % str(out.shape)


def test_forward_finite():
    import torch

    m = BERTClassifier(config={"train_fresh": True})
    ids = torch.zeros(1, 8, dtype=torch.long)
    mask = torch.ones(1, 8, dtype=torch.long)
    with torch.no_grad():
        out = m(ids, mask)
    assert torch.isfinite(out).all()


def test_has_dropout(model):
    assert hasattr(model, "dropout")
