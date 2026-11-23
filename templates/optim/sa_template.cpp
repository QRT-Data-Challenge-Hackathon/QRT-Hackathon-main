// Simulated-annealing template in C++ (use when Python is too slow for the time limit).
// Build: g++ -O2 -std=c++17 sa_template.cpp -o sa && ./sa < input.txt
// Fill in: read input, initial (VALID) solution, delta-score of a move, output.
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using namespace std;
using Clock = chrono::steady_clock;

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n; if (!(cin >> n)) return 0;
    vector<long long> v(n); for (auto& x : v) cin >> x;

    vector<int> sol(n); iota(sol.begin(), sol.end(), 0);           // trivial valid answer
    auto score = [&](const vector<int>& s) {                         // judge's formula
        long long t = 0; for (int i = 0; i < n; ++i) t += 1LL * i * v[s[i]]; return t; };

    mt19937_64 rng(12345);
    const double BUDGET = 1.8, T0 = 10.0, T1 = 0.01;               // seconds, temperatures
    long long cur = score(sol), best = cur; vector<int> bestSol = sol;
    auto start = Clock::now();
    for (long long it = 0;; ++it) {
        double el = chrono::duration<double>(Clock::now() - start).count();
        if (el > BUDGET || n < 2) break;
        double T = T0 * pow(T1 / T0, el / BUDGET);
        int i = rng() % n, j = rng() % n; if (i == j) continue;
        // O(1) delta for a swap; replace with your problem's incremental delta
        long long d = 1LL * i * (v[sol[j]] - v[sol[i]]) + 1LL * j * (v[sol[i]] - v[sol[j]]);
        if (d >= 0 || exp(d / T) > (rng() % 1000000) / 1e6) {
            swap(sol[i], sol[j]); cur += d;
            if (cur > best) { best = cur; bestSol = sol; }
        }
    }
    for (int i = 0; i < n; ++i) cout << bestSol[i] << (i + 1 < n ? ' ' : '\n');
}
