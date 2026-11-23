"""Fallback kit for QRT's "C3 — Code. Compute. Compete." format (HackerRank optimisation
problems with partial-credit scoring). Generic, time-budgeted search engines:

  * :func:`anneal`        simulated annealing over any state with a neighbour move
  * :func:`hill_climb`    first-improvement local search with random restarts
  * :func:`beam_search`   constructive search keeping the ``width`` best partial states
  * :func:`noisy_minimise` black-box minimisation of a noisy function (repeat + average)

Pattern that won C3 events: valid (even trivial) answer on every problem first, then a local
scorer identical to the judge's, then search under a wall-clock budget.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar("S")


@dataclass(frozen=True)
class SearchResult(Generic[S]):
    state: S
    score: float
    iterations: int
    elapsed_s: float


def anneal(initial: S, score: Callable[[S], float], neighbour: Callable[[S, random.Random], S],
           budget_s: float = 5.0, t_start: float = 1.0, t_end: float = 1e-3,
           maximise: bool = True, seed: int = 0) -> SearchResult[S]:
    """Simulated annealing with geometric cooling over the wall-clock budget.

    ``neighbour`` must return a NEW state (never mutate its input)."""
    rng, sign = random.Random(seed), 1.0 if maximise else -1.0
    cur, cur_s = initial, sign * score(initial)
    best, best_s = cur, cur_s
    start, it = time.perf_counter(), 0
    while (elapsed := time.perf_counter() - start) < budget_s:
        temp = t_start * (t_end / t_start) ** (elapsed / budget_s)
        cand = neighbour(cur, rng)
        cand_s = sign * score(cand)
        if cand_s >= cur_s or rng.random() < math.exp((cand_s - cur_s) / max(temp, 1e-12)):
            cur, cur_s = cand, cand_s
            if cur_s > best_s:
                best, best_s = cur, cur_s
        it += 1
    return SearchResult(best, sign * best_s, it, time.perf_counter() - start)


def hill_climb(make_initial: Callable[[random.Random], S], score: Callable[[S], float],
               neighbour: Callable[[S, random.Random], S], budget_s: float = 5.0,
               patience: int = 500, maximise: bool = True, seed: int = 0) -> SearchResult[S]:
    """Local search; restarts from a fresh initial state after ``patience`` failed moves."""
    rng, sign = random.Random(seed), 1.0 if maximise else -1.0
    best, best_s, it = None, -math.inf, 0
    start = time.perf_counter()
    while time.perf_counter() - start < budget_s:
        cur = make_initial(rng)
        cur_s, fails = sign * score(cur), 0
        while fails < patience and time.perf_counter() - start < budget_s:
            cand = neighbour(cur, rng)
            cand_s = sign * score(cand)
            it += 1
            cur, cur_s, fails = (cand, cand_s, 0) if cand_s > cur_s else (cur, cur_s, fails + 1)
        if cur_s > best_s:
            best, best_s = cur, cur_s
    return SearchResult(best, sign * best_s, it, time.perf_counter() - start)


def beam_search(initial: S, expand: Callable[[S], Iterable[S]], score: Callable[[S], float],
                is_complete: Callable[[S], bool], width: int = 50,
                budget_s: float = 5.0) -> SearchResult[S]:
    """Keep the ``width`` best partial states per depth (higher score is better)."""
    start, beam, it = time.perf_counter(), [initial], 0
    best, best_s = None, -math.inf
    while beam and time.perf_counter() - start < budget_s:
        children = [c for s in beam for c in expand(s)]
        it += len(children)
        for child in (c for c in children if is_complete(c)):
            if (s := score(child)) > best_s:
                best, best_s = child, s
        partial = [c for c in children if not is_complete(c)]
        beam = sorted(partial, key=score, reverse=True)[:width]
    return SearchResult(best, best_s, it, time.perf_counter() - start)


def noisy_minimise(f: Callable[[list[float]], float], x0: list[float], step: float = 0.5,
                   repeats: int = 5, budget_evals: int = 2000, seed: int = 0
                   ) -> SearchResult[list[float]]:
    """(1+1)-ES with 1/5th success rule on the mean of ``repeats`` noisy evaluations."""
    rng = random.Random(seed)
    start, evals = time.perf_counter(), 0

    def mean_f(x: list[float]) -> float:
        nonlocal evals
        evals += repeats
        return sum(f(x) for _ in range(repeats)) / repeats

    x, fx = list(x0), mean_f(x0)
    while evals + repeats <= budget_evals:
        cand = [xi + rng.gauss(0, step) for xi in x]
        fc = mean_f(cand)
        if fc < fx:
            x, fx, step = cand, fc, step * 1.5
        else:
            step *= 1.5 ** -0.25
    return SearchResult(x, fx, evals, time.perf_counter() - start)
