from __future__ import annotations

import math


def precision_recall_f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    if not expected:
        return 0.0, 1.0, 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, index in enumerate(ordered, start=1):
        ranks[index] = float(rank)
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    return round(_pearson(_ranks(xs), _ranks(ys)), 3)


def pairwise_ranking_accuracy(scores: list[float], labels: list[str]) -> float | None:
    goods = [s for s, lab in zip(scores, labels, strict=True) if lab == "Good Fit"]
    bads = [s for s, lab in zip(scores, labels, strict=True) if lab == "No Fit"]
    if not goods or not bads:
        return None
    wins = sum(1 for g in goods for b in bads if g > b)
    return round(wins / (len(goods) * len(bads)), 3)


def bin_label(score: float) -> str:
    if score < 40:
        return "No Fit"
    if score < 65:
        return "Potential Fit"
    return "Good Fit"


def label_accuracy(scores: list[float], labels: list[str]) -> float | None:
    usable = [(s, lab) for s, lab in zip(scores, labels, strict=True) if lab]
    if not usable:
        return None
    correct = sum(1 for s, lab in usable if bin_label(s) == lab)
    return round(correct / len(usable), 3)


LABEL_ORDINAL = {"No Fit": 0.0, "Potential Fit": 1.0, "Good Fit": 2.0}
