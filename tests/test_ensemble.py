import sys, os, pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.ensemble_detector import EnsembleDetector


@pytest.fixture
def ens():
    # Use a detector with all base models set to None; predict should still
    # return a well-formed neutral output without raising.
    return EnsembleDetector(
        bert_model=None,
        tfidf_model=None,
        nb_model=None,
        weights=None,
    )


REQUIRED = [
    "crisp_label",
    "ensemble_probability",
    "model_breakdown",
    "ensemble_weights",
    "model_agreement",
]


def test_required_keys(ens):
    r = ens.predict("Test article.")
    for k in REQUIRED:
        assert k in r, "Missing: %s" % k


def test_crisp_label_valid(ens):
    r = ens.predict("Test.")
    assert r["crisp_label"] in ("credible", "misinformation")


def test_probability_in_range(ens):
    r = ens.predict("Test.")
    assert 0.0 <= r["ensemble_probability"] <= 1.0


def test_breakdown_has_all_models(ens):
    r = ens.predict("Test.")
    for m in ["bert", "tfidf", "naive_bayes"]:
        assert m in r["model_breakdown"]


def test_agreement_in_range(ens):
    r = ens.predict("Test.")
    assert 0.0 <= r["model_agreement"] <= 1.0


def test_weights_sum_to_one(ens):
    r = ens.predict("Test.")
    assert abs(sum(r["ensemble_weights"].values()) - 1.0) < 0.01


def test_varied_inputs_no_crash(ens):
    for t in ["Short.", "Word " * 100, "", "123"]:
        assert "crisp_label" in ens.predict(t)
