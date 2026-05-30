"""Контракт признаков матч-классификатора Uneemi (775d).

Почему так устроено:
- Профиль описывается вектором `profile_vec` (775d) = `board_emb` (768d, усреднённый
  и L2-нормированный выход SigLIP по доске пользователя) + `quiz` (7d, ответы анкеты).
  Именно этот 775d-вектор кладётся в фичестор Feast (768 board + 7 quiz).
- Классификатор матча принимает признаки ПАРЫ профилей `pair_features` (775d):
  поэлементное произведение board-частей (768d, ловит взаимную близость стиля) и
  модуль разности quiz-частей (7d, ловит расхождение по анкете). Произведение и
  разность симметричны/полусимметричны - не зависят от порядка A,B так, чтобы
  модель не выучивала искусственную асимметрию.
- Размерность 775 одинакова у профиля и у признаков пары - удобно объяснять на защите.
"""

from __future__ import annotations

import numpy as np

from uneemi_ml.config import EMBED_DIM

# Названия quiz-признаков. Порядок фиксирован - он же порядок колонок в Feast и
# в обучающих матрицах. Менять только синхронно с feature_repo и обучением.
QUIZ_FEATURES: tuple[str, ...] = (
    "age_norm",
    "extraversion",
    "openness",
    "activity_level",
    "aesthetics_pref",
    "social_pref",
    "risk_pref",
)
QUIZ_DIM: int = len(QUIZ_FEATURES)  # 7
PROFILE_DIM: int = EMBED_DIM + QUIZ_DIM  # 775
PAIR_DIM: int = EMBED_DIM + QUIZ_DIM  # 775

_EPS: float = 1e-8


def l2_normalize(vec: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-нормировка. Нормируем, чтобы косинус сводился к скалярному произведению."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return vec / np.maximum(norm, _EPS)


def board_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Свернуть доску (N, 768) в один board-вектор (768,): среднее + L2-норма.

    Среднее по доске - устойчивое представление «общего вайба»; L2-норма делает
    последующие косинусные сравнения сопоставимыми между профилями.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"board_centroid: ожидался массив (N, {EMBED_DIM}), получено {embeddings.shape}"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("board_centroid: пустая доска (0 изображений)")
    centroid = embeddings.mean(axis=0)
    return l2_normalize(centroid).astype(np.float32)


def build_profile_vector(board_emb: np.ndarray, quiz: np.ndarray) -> np.ndarray:
    """Собрать profile_vec (775,) = board_emb (768) ++ quiz (7)."""
    board_emb = np.asarray(board_emb, dtype=np.float32).reshape(-1)
    quiz = np.asarray(quiz, dtype=np.float32).reshape(-1)
    if board_emb.shape[0] != EMBED_DIM:
        raise ValueError(f"build_profile_vector: board_emb должен быть {EMBED_DIM}d")
    if quiz.shape[0] != QUIZ_DIM:
        raise ValueError(f"build_profile_vector: quiz должен быть {QUIZ_DIM}d")
    return np.concatenate([board_emb, quiz]).astype(np.float32)


def split_profile_vector(profile_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Разобрать profile_vec (775,) обратно на (board_emb 768, quiz 7)."""
    profile_vec = np.asarray(profile_vec, dtype=np.float32).reshape(-1)
    if profile_vec.shape[0] != PROFILE_DIM:
        raise ValueError(f"split_profile_vector: ожидался {PROFILE_DIM}d вектор")
    return profile_vec[:EMBED_DIM], profile_vec[EMBED_DIM:]


def build_pair_features(profile_a: np.ndarray, profile_b: np.ndarray) -> np.ndarray:
    """Признаки пары (775,): board_A*board_B (768) ++ |quiz_A - quiz_B| (7)."""
    board_a, quiz_a = split_profile_vector(profile_a)
    board_b, quiz_b = split_profile_vector(profile_b)
    board_interaction = board_a * board_b
    quiz_distance = np.abs(quiz_a - quiz_b)
    return np.concatenate([board_interaction, quiz_distance]).astype(np.float32)


def build_pair_matrix(profiles_a: np.ndarray, profiles_b: np.ndarray) -> np.ndarray:
    """Векторно собрать матрицу признаков пар (N, 775) из (N, 775) и (N, 775)."""
    profiles_a = np.asarray(profiles_a, dtype=np.float32)
    profiles_b = np.asarray(profiles_b, dtype=np.float32)
    if profiles_a.shape != profiles_b.shape or profiles_a.shape[1] != PROFILE_DIM:
        raise ValueError(
            f"build_pair_matrix: ожидались две матрицы (N, {PROFILE_DIM}), "
            f"получено {profiles_a.shape} и {profiles_b.shape}"
        )
    board_a, quiz_a = profiles_a[:, :EMBED_DIM], profiles_a[:, EMBED_DIM:]
    board_b, quiz_b = profiles_b[:, :EMBED_DIM], profiles_b[:, EMBED_DIM:]
    board_interaction = board_a * board_b
    quiz_distance = np.abs(quiz_a - quiz_b)
    return np.concatenate([board_interaction, quiz_distance], axis=1).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов (используется в честной генерации меток)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < _EPS:
        return 0.0
    return float(np.dot(a, b) / denom)
