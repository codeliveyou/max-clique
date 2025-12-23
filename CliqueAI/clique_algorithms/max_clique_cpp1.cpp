#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace py = pybind11;

static inline int popcnt64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    // fallback
    int c = 0;
    while (x) { x &= (x - 1); ++c; }
    return c;
#endif
}

struct Bitset {
    int n = 0;
    int blocks = 0;
    std::vector<uint64_t> w;

    Bitset() {}
    Bitset(int n_) { reset_size(n_); }

    void reset_size(int n_) {
        n = n_;
        blocks = (n + 63) / 64;
        w.assign(blocks, 0ULL);
    }

    inline void set(int i) { w[i >> 6] |= (1ULL << (i & 63)); }
    inline void reset(int i) { w[i >> 6] &= ~(1ULL << (i & 63)); }
    inline bool test(int i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }

    inline bool empty() const {
        for (auto x : w) if (x) return false;
        return true;
    }

    inline int count() const {
        int s = 0;
        for (auto x : w) s += popcnt64(x);
        return s;
    }

    inline void intersect_with(const Bitset& other) {
        for (int i = 0; i < blocks; ++i) w[i] &= other.w[i];
    }

    inline Bitset intersection(const Bitset& other) const {
        Bitset r;
        r.n = n;
        r.blocks = blocks;
        r.w.resize(blocks);
        for (int i = 0; i < blocks; ++i) r.w[i] = w[i] & other.w[i];
        return r;
    }

    // remove all bits of other: this &= ~other
    inline void and_not_with(const Bitset& other) {
        for (int i = 0; i < blocks; ++i) w[i] &= ~other.w[i];
    }

    // Return any set bit index, or -1
    inline int any_one() const {
        for (int b = 0; b < blocks; ++b) {
            uint64_t x = w[b];
            if (!x) continue;
#if defined(__GNUG__) || defined(__clang__)
            int bit = __builtin_ctzll(x);
#else
            int bit = 0;
            while (((x >> bit) & 1ULL) == 0ULL) ++bit;
#endif
            int idx = (b << 6) + bit;
            return (idx < n) ? idx : -1;
        }
        return -1;
    }

    // Extract all indices in descending order (often helpful)
    inline void to_vec(std::vector<int>& out) const {
        out.clear();
        out.reserve(count());
        for (int i = 0; i < n; ++i) if (test(i)) out.push_back(i);
    }
};

struct MaxCliqueSolver {
    int n = 0;
    int blocks = 0;
    std::vector<Bitset> adj;          // adjacency bitsets
    std::vector<int> deg;             // degrees

    std::chrono::steady_clock::time_point deadline;
    std::atomic<int> best_size;
    std::vector<int> best_clique;
    std::mutex best_mtx;

    MaxCliqueSolver(int n_, const std::vector<std::vector<int>>& adjacency_list)
        : n(n_) {
        blocks = (n + 63) / 64;
        adj.assign(n, Bitset(n));
        deg.assign(n, 0);

        // Build symmetric bitset adjacency (robust if input is asymmetric)
        for (int u = 0; u < n; ++u) {
            if (u >= (int)adjacency_list.size()) break;
            for (int v : adjacency_list[u]) {
                if (v < 0 || v >= n || v == u) continue;
                adj[u].set(v);
                adj[v].set(u);
            }
        }
        for (int u = 0; u < n; ++u) deg[u] = adj[u].count();
    }

    inline bool time_up() const {
        return std::chrono::steady_clock::now() >= deadline;
    }

    // Fast greedy clique to set a good initial incumbent
    std::vector<int> greedy_incumbent() const {
        std::vector<int> order(n);
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return deg[a] > deg[b];
        });

        std::vector<int> clique;
        Bitset cand(n);
        for (int i = 0; i < n; ++i) cand.set(i);

        for (int v : order) {
            if (!cand.test(v)) continue;
            clique.push_back(v);
            cand.intersect_with(adj[v]);
            if (cand.empty()) break;
        }
        return clique;
    }

    // Greedy sequential coloring upper bound on P.
    // Returns a pair: (vertices ordered for branching, color_bound per vertex)
    // Classic Tomita style: produce an ordering such that later vertices have <= bound.
    void color_sort(const Bitset& P, std::vector<int>& verts, std::vector<int>& colors) const {
        std::vector<int> U;
        P.to_vec(U);

        // Sort candidates by degree (in original graph) for better coloring
        std::sort(U.begin(), U.end(), [&](int a, int b) { return deg[a] > deg[b]; });

        verts.clear();
        colors.clear();
        verts.reserve(U.size());
        colors.reserve(U.size());

        // Color classes built as independent sets in the subgraph induced by U.
        // We implement: while U not empty: create new color class, greedily add vertices
        int color = 0;
        std::vector<int> remaining = U;

        while (!remaining.empty()) {
            ++color;
            Bitset forbidden(n);
            std::vector<int> next_remaining;
            next_remaining.reserve(remaining.size());

            for (int v : remaining) {
                // v can be added to this color class if it's not forbidden
                if (!forbidden.test(v)) {
                    verts.push_back(v);
                    colors.push_back(color);
                    // forbid neighbors of v (so we keep this class independent)
                    for (int b = 0; b < blocks; ++b) forbidden.w[b] |= adj[v].w[b];
                } else {
                    next_remaining.push_back(v);
                }
            }
            remaining.swap(next_remaining);
        }

        // We branch from the end (largest color numbers last) in Tomita;
        // but pruning checks use colors[i] as upper bound.
        // Ensure verts/colors aligned.
    }

    void update_best(const std::vector<int>& clique) {
        int sz = (int)clique.size();
        int cur_best = best_size.load(std::memory_order_relaxed);
        if (sz <= cur_best) return;

        // Try fast CAS loop first
        while (sz > cur_best) {
            if (best_size.compare_exchange_weak(cur_best, sz, std::memory_order_relaxed)) {
                std::lock_guard<std::mutex> lk(best_mtx);
                best_clique = clique;
                return;
            }
        }
    }

    // Recursive expansion (Tomita-style)
    void expand(Bitset P, std::vector<int>& C) {
        if (time_up()) return;

        // Upper bound via coloring
        std::vector<int> verts, col;
        color_sort(P, verts, col);

        // Branch in reverse order (common in Tomita MCQ)
        for (int i = (int)verts.size() - 1; i >= 0; --i) {
            if (time_up()) return;

            int v = verts[i];
            int ub = (int)C.size() + col[i];
            if (ub <= best_size.load(std::memory_order_relaxed)) return; // prunes all earlier (smaller col) too

            // Choose v
            C.push_back(v);
            Bitset P2 = P.intersection(adj[v]);

            if (P2.empty()) {
                update_best(C);
            } else {
                // Cheap bound: |C| + |P2| <= best => prune
                if ((int)C.size() + P2.count() > best_size.load(std::memory_order_relaxed)) {
                    expand(P2, C);
                }
            }

            // Backtrack
            C.pop_back();
            P.reset(v);
        }
    }

    std::vector<int> solve(double time_limit_sec, int threads) {
        deadline = std::chrono::steady_clock::now()
         + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
               std::chrono::duration<double>(time_limit_sec)
           );


        // Initial incumbent
        best_clique = greedy_incumbent();
        best_size.store((int)best_clique.size(), std::memory_order_relaxed);

        // Root candidate set: all vertices
        Bitset P(n);
        for (int i = 0; i < n; ++i) P.set(i);

        // Create a good branching order for top-level split
        std::vector<int> root_verts, root_col;
        color_sort(P, root_verts, root_col);

        // We’ll split the last K branches (highest colors) across threads
        // (Parallelizing deeper typically hurts due to overhead/locking.)
        if (threads < 1) threads = 1;
        int K = std::min((int)root_verts.size(), threads * 2); // a bit more tasks than threads

        std::vector<std::thread> pool;
        pool.reserve(threads);

        std::atomic<int> next_task(0);

        auto worker = [&]() {
            while (!time_up()) {
                int t = next_task.fetch_add(1);
                if (t >= K) break;

                int idx = (int)root_verts.size() - 1 - t;
                if (idx < 0) break;

                int v = root_verts[idx];

                std::vector<int> C;
                C.reserve(n);
                C.push_back(v);

                Bitset P2 = P.intersection(adj[v]);
                if (!P2.empty()) {
                    if ((int)C.size() + P2.count() > best_size.load(std::memory_order_relaxed)) {
                        expand(P2, C);
                    }
                } else {
                    update_best(C);
                }
            }
        };

        int spawn = std::min(threads, 32);
        for (int i = 0; i < spawn; ++i) pool.emplace_back(worker);
        for (auto& th : pool) th.join();

        // If time remains, continue single-threaded on remaining branches for completeness
        if (!time_up()) {
            std::vector<int> C;
            expand(P, C);
        }

        // Return best clique found
        std::lock_guard<std::mutex> lk(best_mtx);
        std::sort(best_clique.begin(), best_clique.end());
        return best_clique;
    }
};

std::vector<int> max_clique_cpp1(int number_of_nodes,
                               const std::vector<std::vector<int>>& adjacency_list,
                               double time_limit_sec = 29.5,
                               int threads = 8) {
    MaxCliqueSolver solver(number_of_nodes, adjacency_list);
    return solver.solve(time_limit_sec, threads);
}

PYBIND11_MODULE(max_clique_cpp1, m) {
    m.doc() = "High-performance maximum clique (bitset BnB + coloring bound + top-level parallelism)";
    m.def("max_clique_cpp1", &max_clique_cpp1,
          py::arg("number_of_nodes"),
          py::arg("adjacency_list"),
          py::arg("time_limit_sec") = 29.5,
          py::arg("threads") = 8);
}

/*
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  max_clique_cpp1.cpp \
  -o max_clique_cpp1$(python3-config --extension-suffix)
*/