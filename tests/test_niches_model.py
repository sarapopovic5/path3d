"""Guards on the vendored niche classifier bundle.

The bundle was pickled with an older scikit-learn than is necessarily
installed, and sklearn warns that unpickling across versions "might lead to
breaking code or invalid results". These tests replace that warning with an
actual check: the estimator is a plain ``LogisticRegression``, so its output is
a closed-form softmax of ``coef_ @ x + intercept_``, and we can verify the
unpickled object still computes exactly that under whatever sklearn is present.

If a future scikit-learn ever does break the unpickle, this fails loudly here
instead of silently degrading every prediction in a multi-hour stack run.
"""

from __future__ import annotations

import numpy as np
import pytest

from path3d.config import NICHE_LABEL_INDEX
from path3d.niches.predict import DEFAULT_MODEL_PATH, load_model

EXPECTED_CLASSES = ["acellular", "epithelium", "immune", "stroma"]
EMBED_DIM = 1536


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_packaged_bundle_is_present():
    """The wheel must ship the classifier; it is the only non-code asset."""
    assert DEFAULT_MODEL_PATH.exists(), (
        f"packaged bundle missing at {DEFAULT_MODEL_PATH}"
    )


def test_classes_match_the_label_index(model):
    """Every class the model can emit must have a label index to rasterise to."""
    assert model.classes == EXPECTED_CLASSES
    for name in model.classes:
        assert name in NICHE_LABEL_INDEX


def test_grid_geometry_comes_from_the_bundle(model):
    """32 um tiles in 112 um windows -- what the model card documents."""
    assert model.tile_um == 32.0
    assert model.fov_um == 112.0


def test_unpickled_estimator_still_computes_the_closed_form(model):
    """The real answer to sklearn's InconsistentVersionWarning.

    LogisticRegression's multinomial ``predict_proba`` is exactly
    ``softmax(X @ coef_.T + intercept_)``. If the unpickle lost or misread any
    fitted attribute, this diverges.
    """
    clf = model.clf
    assert clf.coef_.shape == (len(EXPECTED_CLASSES), EMBED_DIM)
    assert clf.intercept_.shape == (len(EXPECTED_CLASSES),)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, EMBED_DIM))
    x /= np.linalg.norm(x, axis=1, keepdims=True)  # the model expects L2-normalised

    z = x @ clf.coef_.T + clf.intercept_
    e = np.exp(z - z.max(axis=1, keepdims=True))
    expected = e / e.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(clf.predict_proba(x), expected, atol=1e-12)


def test_probabilities_are_well_formed(model):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, EMBED_DIM))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    p = model.clf.predict_proba(x)
    assert p.shape == (32, len(EXPECTED_CLASSES))
    assert np.isfinite(p).all()
    assert (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-10)


def test_validation_metrics_are_carried_for_reporting(model):
    """The model card's headline numbers travel with the weights."""
    assert set(model.auc_by_class) == set(EXPECTED_CLASSES)
    # Epithelium and immune are the dependable classes per the model card.
    assert model.auc_by_class["epithelium"] > 0.9
    assert model.auc_by_class["immune"] > 0.9


def test_classify_embeddings_orders_columns_by_classes(model):
    """Column order must follow model.classes, not insertion or sort order."""
    from path3d.niches.predict import call_columns, classify_embeddings, prob_columns

    rng = np.random.default_rng(2)
    emb = rng.normal(size=(64, EMBED_DIM)).astype(np.float16)

    p, cols = classify_embeddings(emb, model, quantile=0.80)

    for k, name in enumerate(prob_columns(model.classes)):
        np.testing.assert_allclose(cols[name].to_numpy(), p[:, k], rtol=1e-6)
    # A top-20% call on 64 tiles selects ~13 per class.
    for name in call_columns(model.classes):
        assert 5 <= int(cols[name].sum()) <= 20
    assert set(cols["argmax_class"]) <= set(model.classes)
