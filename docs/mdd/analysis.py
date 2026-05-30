"""MDD-анализ времени отклика системы: репродукция графика ТЗ + статистика.

Воспроизводит ровно тот сэмплинг и визуализацию, что даны в задании (два набора
latency: существующая система loc=3.5, улучшенная loc=2.0, scale=0.4, n=500000,
seed=42), и доводит до архитектурного решения по MDD.

Зрелость анализа: при n=500000 любой статтест даёт p ~ 0 (статзначимость
гарантирована размером выборки), поэтому решение принимается по ПРАКТИЧЕСКОЙ
значимости - размер эффекта Cohen's d, доверительный интервал разности средних и,
главное, пересечение SLO по времени отклика.

Запуск: uv run python docs/mdd/analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея - сохраняем в файл
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

OUT_PNG = Path(__file__).resolve().parent / "latency_comparison.png"
SLO_RESPONSE_SEC = 3.0  # бизнес-SLO: время отклика p95 не выше 3 секунд


def make_data() -> tuple[np.ndarray, np.ndarray]:
    """Ровно как в ТЗ: фиксированный seed и параметры распределений."""
    np.random.seed(42)
    existing_system_responses = np.random.normal(loc=3.5, scale=0.4, size=500000)
    improved_system_responses = np.random.normal(loc=2.0, scale=0.4, size=500000)
    return existing_system_responses, improved_system_responses


def plot(existing: np.ndarray, improved: np.ndarray) -> None:
    """Репродукция графика ТЗ один-в-один (KDE двух распределений)."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(existing, label="Существующая система", fill=True, color="red")
    sns.kdeplot(improved, label="Улучшенная система", fill=True, color="green")
    plt.title("Сравнение времени отклика системы")
    plt.xlabel("Время отклика (секунды)")
    plt.ylabel("Наблюдения")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    new_yticks = np.linspace(ymin, ymax, 5)
    new_yticklabels = [f"{int((t / ymax) * 100)}%" if ymax != 0 else "0%" for t in new_yticks]
    ax.set_yticks(new_yticks)
    ax.set_yticklabels(new_yticklabels)
    plt.axvline(SLO_RESPONSE_SEC, color="black", linestyle=":", label="SLO 3.0с")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    plt.close()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d с объединённым стандартным отклонением (размер эффекта)."""
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def ci_mean_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """95% доверительный интервал разности средних (Welch)."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(diff - z * se), float(diff + z * se)


def main() -> int:
    existing, improved = make_data()
    plot(existing, improved)

    # Гипотезы: H0 - средние времена отклика равны; H1 - различаются (двусторонний).
    alpha = 0.05
    t_stat, t_p = stats.ttest_ind(existing, improved, equal_var=False)  # Welch
    u_stat, u_p = stats.mannwhitneyu(existing, improved, alternative="two-sided")
    d = cohens_d(existing, improved)
    ci_low, ci_high = ci_mean_diff(existing, improved, alpha)
    h0_rejected = t_p < alpha

    p95_exist = float(np.percentile(existing, 95))
    p95_impr = float(np.percentile(improved, 95))
    above_exist = float((existing > SLO_RESPONSE_SEC).mean())
    above_impr = float((improved > SLO_RESPONSE_SEC).mean())

    print("=== MDD-анализ времени отклика ===")
    print(f"Среднее: существующая={existing.mean():.4f}с, улучшенная={improved.mean():.4f}с")
    print(f"Разность средних: {existing.mean() - improved.mean():.4f}с")
    verdict = f"H0 отклонена при alpha={alpha}" if h0_rejected else "H0 не отклонена"
    print(f"Welch t-test: t={t_stat:.2f}, p={t_p:.3e}; {verdict}")
    print(f"Mann-Whitney U: U={u_stat:.0f}, p={u_p:.3e}")
    print(f"Cohen's d: {d:.3f}")
    print(f"95% ДИ разности средних: [{ci_low:.4f}, {ci_high:.4f}] с")
    print(f"p95 (SLO={SLO_RESPONSE_SEC}с): сущ={p95_exist:.3f}с, улучш={p95_impr:.3f}с")
    print(f"Доля выше SLO: существующая={above_exist:.3%}, улучшенная={above_impr:.3%}")
    print(f"График сохранён: {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
