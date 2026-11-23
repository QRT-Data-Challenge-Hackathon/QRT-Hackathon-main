import random

from qrtc import optim


def _tsp():
    rng = random.Random(1)
    pts = [(rng.random(), rng.random()) for _ in range(25)]

    def length(order):
        return sum(((pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2) ** 0.5
                   for a, b in zip(order, order[1:] + order[:1], strict=True))

    def two_opt(order, r):
        i, j = sorted(r.sample(range(len(order)), 2))
        return order[:i] + order[i:j + 1][::-1] + order[j + 1:]

    return list(range(25)), length, two_opt


def test_anneal_improves_tour():
    init, length, move = _tsp()
    res = optim.anneal(init, length, move, budget_s=0.5, maximise=False, t_start=0.3)
    assert res.score < length(init) * 0.7
    assert res.state != init and sorted(res.state) == init


def test_hill_climb_restarts():
    init, length, move = _tsp()
    res = optim.hill_climb(lambda r: r.sample(init, len(init)), length, move, budget_s=0.3,
                           maximise=False)
    assert sorted(res.state) == init and res.iterations > 0


def test_beam_search_builds_best_bitstring():
    res = optim.beam_search((), lambda s: [s + (0,), s + (1,)], lambda s: sum(s),
                            lambda s: len(s) == 8, width=4, budget_s=1)
    assert res.state == (1,) * 8


def test_noisy_minimise_finds_quadratic_minimum():
    rng = random.Random(0)
    res = optim.noisy_minimise(lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2 + rng.gauss(0, 0.05),
                               [0.0, 0.0], budget_evals=3000)
    assert abs(res.state[0] - 3) < 0.3 and abs(res.state[1] + 1) < 0.3
