"""Юнит-тесты контракта признаков (775d). Чистый numpy, без ONNX и pandas."""

from __future__ import annotations

import numpy as np
import pytest

from uneemi_ml.features import (
    EMBED_DIM,
    PAIR_DIM,
    PROFILE_DIM,
    QUIZ_DIM,
    board_centroid,
    build_pair_features,
    build_pair_matrix,
    build_profile_vector,
    cosine,
    l2_normalize,
    split_profile_vector,
)


def test_dims_consistent() -> None:
    """775 = 768 + 7 и для профиля, и для признаков пары."""
    assert QUIZ_DIM == 7
    assert PROFILE_DIM == EMBED_DIM + QUIZ_DIM == 775
    assert PAIR_DIM == 775


def test_build_and_split_roundtrip() -> None:
    rng = np.random.default_rng(0)
    board = rng.normal(size=EMBED_DIM).astype(np.float32)
    quiz = rng.uniform(size=QUIZ_DIM).astype(np.float32)
    vec = build_profile_vector(board, quiz)
    assert vec.shape == (PROFILE_DIM,)
    back_board, back_quiz = split_profile_vector(vec)
    assert np.allclose(back_board, board)
    assert np.allclose(back_quiz, quiz)


def test_board_centroid_is_unit_norm() -> None:
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(10, EMBED_DIM)).astype(np.float32)
    c = board_centroid(emb)
    assert c.shape == (EMBED_DIM,)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(c))


def test_board_centroid_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        board_centroid(np.zeros((0, EMBED_DIM), dtype=np.float32))
    with pytest.raises(ValueError):
        board_centroid(np.zeros((3, EMBED_DIM + 1), dtype=np.float32))


def test_pair_features_shape_and_symmetry() -> None:
    """board-часть (произведение) симметрична по A,B; quiz-часть (|diff|) тоже."""
    rng = np.random.default_rng(2)
    a = build_profile_vector(rng.normal(size=EMBED_DIM), rng.uniform(size=QUIZ_DIM))
    b = build_profile_vector(rng.normal(size=EMBED_DIM), rng.uniform(size=QUIZ_DIM))
    pf_ab = build_pair_features(a, b)
    pf_ba = build_pair_features(b, a)
    assert pf_ab.shape == (PAIR_DIM,)
    assert np.allclose(pf_ab, pf_ba)


def test_build_pair_matrix_matches_rowwise() -> None:
    rng = np.random.default_rng(3)
    pa = np.stack([build_profile_vector(rng.normal(size=EMBED_DIM), rng.uniform(size=QUIZ_DIM))
                   for _ in range(5)])
    pb = np.stack([build_profile_vector(rng.normal(size=EMBED_DIM), rng.uniform(size=QUIZ_DIM))
                   for _ in range(5)])
    mat = build_pair_matrix(pa, pb)
    assert mat.shape == (5, PAIR_DIM)
    for i in range(5):
        assert np.allclose(mat[i], build_pair_features(pa[i], pb[i]))


def test_cosine_bounds_and_self() -> None:
    rng = np.random.default_rng(4)
    v = rng.normal(size=PROFILE_DIM).astype(np.float32)
    assert pytest.approx(1.0, abs=1e-5) == cosine(v, v)
    assert cosine(v, -v) < -0.99
    assert cosine(np.zeros(PROFILE_DIM), v) == 0.0


def test_l2_normalize_unit() -> None:
    rng = np.random.default_rng(5)
    v = rng.normal(size=PROFILE_DIM).astype(np.float32)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(l2_normalize(v)))
