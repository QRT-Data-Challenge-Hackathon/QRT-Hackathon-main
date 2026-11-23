"""HackerRank-style optimisation solution template (QRT "C3" format fallback).

Workflow (what top C3 teams did):
  1. Paste the problem's I/O format into parse()/emit(). Submit a TRIVIAL valid answer first:
     invalid output scores 0, valid-but-bad output earns partial credit.
  2. Re-implement the judge's score in score() and check it on the sample tests.
  3. Improve with a greedy initial solution + time-budgeted local search / annealing.
  4. Keep BUDGET_S well under the judge's time limit (Python is slow: ~1e6 simple ops/s).
Self-contained on purpose (judges run a single file); the engine mirrors qrtc.optim.anneal.
"""

import math
import random
import sys
import time

BUDGET_S = 1.5  # seconds of search; leave margin for parsing and output


def parse(text: str):
    tokens = text.split()
    n = int(tokens[0])
    values = list(map(int, tokens[1:1 + n]))
    return {"n": n, "values": values}


def initial_solution(problem):
    return list(range(problem["n"]))  # trivial but VALID


def score(problem, sol) -> float:
    return float(sum(i * problem["values"][s] for i, s in enumerate(sol)))  # judge's formula


def neighbour(sol, rng):
    i, j = rng.randrange(len(sol)), rng.randrange(len(sol))
    new = list(sol)
    new[i], new[j] = new[j], new[i]
    return new


def anneal(problem, sol, budget_s, t_start=10.0, t_end=0.01, seed=0):
    rng, start = random.Random(seed), time.perf_counter()
    cur, cur_s = sol, score(problem, sol)
    best, best_s = cur, cur_s
    while (el := time.perf_counter() - start) < budget_s:
        temp = t_start * (t_end / t_start) ** (el / budget_s)
        cand = neighbour(cur, rng)
        cand_s = score(problem, cand)
        if cand_s >= cur_s or rng.random() < math.exp((cand_s - cur_s) / temp):
            cur, cur_s = cand, cand_s
            if cur_s > best_s:
                best, best_s = cur, cur_s
    return best


def emit(sol) -> str:
    return " ".join(map(str, sol))


def main() -> None:
    problem = parse(sys.stdin.read())
    sol = initial_solution(problem)
    if problem["n"] > 1:
        sol = anneal(problem, sol, BUDGET_S)
    sys.stdout.write(emit(sol) + "\n")


if __name__ == "__main__":
    main()
