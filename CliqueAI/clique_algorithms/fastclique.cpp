#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

namespace py = pybind11;

static inline int popcnt_u64(uint64_t x) {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    // portable popcount
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
#endif
}

struct Bits512 {
    // supports up to 512 vertices
    static constexpr int CH = 8; // 8*64=512
    uint64_t w[CH];

    Bits512() { for (int i = 0; i < CH; i++) w[i] = 0; }
    static Bits512 full(int n) {
        Bits512 b;
        int fullChunks = n / 64;
        int rem = n % 64;
        for (int i = 0; i < fullChunks; i++) b.w[i] = ~0ULL;
        if (rem) b.w[fullChunks] = (rem == 64 ? ~0ULL : ((1ULL << rem) - 1ULL));
        return b;
    }
    static Bits512 single(int v) {
        Bits512 b;
        b.w[v >> 6] = 1ULL << (v & 63);
        return b;
    }
    inline bool empty() const {
        for (int i = 0; i < CH; i++) if (w[i]) return false;
        return true;
    }
    inline int popcount() const {
        int s = 0;
        for (int i = 0; i < CH; i++) s += popcnt_u64(w[i]);
        return s;
    }
    inline void AND(const Bits512& o) { for (int i = 0; i < CH; i++) w[i] &= o.w[i]; }
    inline void OR(const Bits512& o) { for (int i = 0; i < CH; i++) w[i] |= o.w[i]; }
    inline void ANDNOT(const Bits512& o) { for (int i = 0; i < CH; i++) w[i] &= ~o.w[i]; }

    inline bool test(int v) const { return (w[v >> 6] >> (v & 63)) & 1ULL; }
    inline void set(int v) { w[v >> 6] |= (1ULL << (v & 63)); }
    inline void reset(int v) { w[v >> 6] &= ~(1ULL << (v & 63)); }

    // iterate: extract one vertex (LSB). Returns -1 if empty.
    inline int pop_lsb() {
        for (int i = 0; i < CH; i++) {
            if (w[i]) {
                uint64_t x = w[i];
#if defined(__GNUG__) || defined(__clang__)
                int b = __builtin_ctzll(x);
#else
                int b = 0; while (((x >> b) & 1ULL) == 0ULL) b++;
#endif
                w[i] &= (w[i] - 1ULL);
                return (i << 6) + b;
            }
        }
        return -1;
    }

    inline Bits512 operator&(const Bits512& o) const {
        Bits512 r;
        for (int i = 0; i < CH; i++) r.w[i] = w[i] & o.w[i];
        return r;
    }
    inline Bits512 operator|(const Bits512& o) const {
        Bits512 r;
        for (int i = 0; i < CH; i++) r.w[i] = w[i] | o.w[i];
        return r;
    }
};

// --------- fast local search (DLS-MC-ish, but in C++ and tight) ----------
struct FastCliqueSolver {
    int n;
    std::vector<Bits512> adj;  // symmetric adjacency bitsets
    std::vector<int> deg;

    // state
    std::vector<uint16_t> miss;   // miss[v] = how many clique vertices v is NOT connected to
    std::vector<uint8_t> inC;
    std::vector<uint16_t> penalty;
    std::vector<uint32_t> tabu_until;

    std::mt19937 rng;
    uint32_t step = 0;

    FastCliqueSolver(int n_, const std::vector<std::vector<int>>& al, uint32_t seed)
        : n(n_), adj(n_), deg(n_, 0),
          miss(n_, 0), inC(n_, 0), penalty(n_, 0), tabu_until(n_, 0),
          rng(seed ? seed : (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()) {

        // build symmetric adjacency
        for (int u = 0; u < n; u++) {
            Bits512 b;
            for (int v : al[u]) {
                if (v >= 0 && v < n && v != u) b.set(v);
            }
            adj[u] = b;
        }
        // symmetrize
        for (int u = 0; u < n; u++) {
            Bits512 tmp = adj[u];
            while (!tmp.empty()) {
                int v = tmp.pop_lsb();
                if (v >= 0 && v < n) adj[v].set(u);
            }
        }
        for (int u = 0; u < n; u++) deg[u] = adj[u].popcount();
    }

    inline int rand_int(int a, int b) {
        std::uniform_int_distribution<int> dist(a, b);
        return dist(rng);
    }
    inline double rand01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(rng);
    }

    Bits512 common_neighbors(const Bits512& Cbits) {
        Bits512 cand = Bits512::full(n);
        Bits512 tmp = Cbits;
        while (!tmp.empty()) {
            int v = tmp.pop_lsb();
            cand.AND(adj[v]);
        }
        cand.ANDNOT(Cbits);
        return cand;
    }

    Bits512 make_maximal(Bits512 Cbits) {
        Bits512 cand = common_neighbors(Cbits);
        while (!cand.empty()) {
            int v = cand.pop_lsb();
            Cbits.set(v);
            cand.AND(adj[v]);
        }
        return Cbits;
    }

    void init_state(const Bits512& Cbits, int Csize) {
        std::fill(inC.begin(), inC.end(), 0);
        for (int i = 0; i < n; i++) miss[i] = 0;

        // mark clique and compute miss
        // miss[v] = |C| - neighbors_in_C
        for (int v = 0; v < n; v++) {
            if (Cbits.test(v)) inC[v] = 1;
        }
        for (int v = 0; v < n; v++) {
            if (inC[v]) { miss[v] = 0; continue; }
            Bits512 inter = adj[v] & Cbits;
            int neighInC = inter.popcount();
            miss[v] = (uint16_t)(Csize - neighInC);
        }
    }

    inline void add_vertex(Bits512& Cbits, int v, int& Csize) {
        inC[v] = 1;
        Cbits.set(v);
        Csize++;

        // update miss for outside vertices that are NOT neighbors of v -> miss++
        for (int u = 0; u < n; u++) {
            if (!inC[u]) {
                if (!adj[v].test(u)) miss[u]++;
            }
        }
        miss[v] = 0;
    }

    inline void remove_vertex(Bits512& Cbits, int v, int& Csize) {
        inC[v] = 0;
        Cbits.reset(v);
        Csize--;

        // update miss for outside vertices that are NOT neighbors of v -> miss--
        for (int u = 0; u < n; u++) {
            if (!inC[u]) {
                if (!adj[v].test(u)) miss[u]--;
            }
        }
    }

    int pick_add_candidate(const Bits512& Cand, int sample_cap) {
        // sample candidates from bitset
        std::vector<int> sample;
        sample.reserve(sample_cap);
        Bits512 tmp = Cand;
        for (int i = 0; i < sample_cap && !tmp.empty(); i++) {
            int v = tmp.pop_lsb();
            sample.push_back(v);
        }
        if (sample.empty()) return -1;
        std::shuffle(sample.begin(), sample.end(), rng);

        int bestv = sample[0];
        long bestScore = -9e18;

        for (int v : sample) {
            if (tabu_until[v] > step) continue;
            long sc = (long)deg[v] * 1000L - (long)penalty[v] * 12L + rand_int(0, 20);
            if (sc > bestScore) { bestScore = sc; bestv = v; }
        }
        return bestv;
    }

    // find swap candidate: u outside with miss[u]==1, drop is unique clique vertex missing with u
    std::pair<int,int> pick_swap_candidate(const Bits512& Cbits, int sample_cap) {
        std::vector<int> outs;
        outs.reserve(sample_cap);

        // collect outside vertices quickly by degree bias: scan all (n~500 ok in C++)
        // keep only miss==1
        for (int u = 0; u < n; u++) {
            if (!inC[u] && miss[u] == 1) outs.push_back(u);
        }
        if (outs.empty()) return {-1,-1};
        std::shuffle(outs.begin(), outs.end(), rng);
        if ((int)outs.size() > sample_cap) outs.resize(sample_cap);

        int bestu = -1, bestdrop = -1;
        long bestScore = -9e18;

        // build clique list once
        std::vector<int> Clist;
        Clist.reserve(256);
        Bits512 tmp = Cbits;
        while (!tmp.empty()) Clist.push_back(tmp.pop_lsb());

        for (int u : outs) {
            if (tabu_until[u] > step) continue;

            int drop = -1;
            for (int v : Clist) {
                if (!adj[u].test(v)) { drop = v; break; }
            }
            if (drop < 0) continue;

            long sc = (long)deg[u] * 1000L - (long)penalty[u] * 14L + (long)penalty[drop] * 18L + rand_int(0, 20);
            if (sc > bestScore) { bestScore = sc; bestu = u; bestdrop = drop; }
        }
        return {bestu, bestdrop};
    }

    Bits512 greedy_start() {
        // pick start among top degrees with bias
        std::vector<int> order(n);
        for (int i = 0; i < n; i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return deg[a] > deg[b]; });

        int topk = std::min(n, 180);
        int idx = (int)((rand01() * rand01()) * topk);
        int v0 = order[idx];

        Bits512 C = Bits512::single(v0);
        Bits512 P = adj[v0];

        // greedy add by "neighbors in P" (fast approximate)
        while (!P.empty()) {
            // choose best v in sampled P
            int bestv = -1;
            int bestsc = -1;

            Bits512 tmp = P;
            int scans = 0;
            while (!tmp.empty() && scans < 256) {
                int v = tmp.pop_lsb();
                scans++;
                int sc = (adj[v] & P).popcount() - (int)penalty[v];
                if (sc > bestsc) { bestsc = sc; bestv = v; }
            }
            if (bestv < 0) break;

            C.set(bestv);
            P.AND(adj[bestv]);
        }
        return make_maximal(C);
    }

    std::vector<int> solve(double time_limit_sec) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto deadline = t0 + std::chrono::duration<double>(time_limit_sec);

        Bits512 bestC;
        int bestSz = 0;

        // parameters tuned for dense ~500
        const int tabu_tenure = 10;
        const int penalty_delay = 220;
        const int sample_add = 220;
        const int sample_swap = 360;

        while (std::chrono::high_resolution_clock::now() < deadline) {
            Bits512 C = greedy_start();
            int Csz = C.popcount();
            init_state(C, Csz);

            if (Csz > bestSz) { bestSz = Csz; bestC = C; }

            int stagnation = 0;

            while (std::chrono::high_resolution_clock::now() < deadline && stagnation < 9000) {
                step++;

                // ADD candidates: miss==0 outside
                Bits512 Cand;
                for (int v = 0; v < n; v++) {
                    if (!inC[v] && miss[v] == 0) Cand.set(v);
                }

                if (!Cand.empty()) {
                    int v = pick_add_candidate(Cand, sample_add);
                    if (v >= 0) {
                        add_vertex(C, v, Csz);
                        C = make_maximal(C); // free gains
                        Csz = C.popcount();
                        init_state(C, Csz);

                        if (Csz > bestSz) { bestSz = Csz; bestC = C; stagnation = 0; }
                        else stagnation++;
                        continue;
                    }
                }

                // SWAP candidates: miss==1
                auto [u, drop] = pick_swap_candidate(C, sample_swap);
                if (u >= 0 && drop >= 0) {
                    remove_vertex(C, drop, Csz);
                    tabu_until[drop] = step + tabu_tenure + rand_int(0, tabu_tenure);
                    add_vertex(C, u, Csz);

                    C = make_maximal(C);
                    Csz = C.popcount();
                    init_state(C, Csz);

                    if (Csz > bestSz) { bestSz = Csz; bestC = C; stagnation = 0; }
                    else stagnation += 3;
                    continue;
                }

                // stuck: penalize clique vertices and kick
                Bits512 tmp = C;
                while (!tmp.empty()) {
                    int v = tmp.pop_lsb();
                    penalty[v]++;
                }
                if (step % penalty_delay == 0) {
                    for (int i = 0; i < n; i++) if (penalty[i] > 0) penalty[i]--;
                }

                // kick 1..3 highest-penalty vertices from clique
                std::vector<int> Clist;
                Clist.reserve(256);
                Bits512 t = C;
                while (!t.empty()) Clist.push_back(t.pop_lsb());
                if (!Clist.empty()) {
                    std::sort(Clist.begin(), Clist.end(), [&](int a, int b){ return penalty[a] > penalty[b]; });
                    int k = (rand01() < 0.50) ? 1 : ((rand01() < 0.85) ? 2 : 3);
                    k = std::min(k, (int)Clist.size());
                    for (int i = 0; i < k; i++) {
                        int v = Clist[i];
                        remove_vertex(C, v, Csz);
                        tabu_until[v] = step + tabu_tenure + rand_int(0, tabu_tenure);
                    }
                    C = make_maximal(C);
                    Csz = C.popcount();
                    init_state(C, Csz);
                }
                stagnation += 25;
            }
        }

        // output best clique as vector<int>
        std::vector<int> out;
        out.reserve(bestSz);
        Bits512 tmp = bestC;
        while (!tmp.empty()) out.push_back(tmp.pop_lsb());
        std::sort(out.begin(), out.end());
        return out;
    }
};


std::vector<int> fastclique_run(int number_of_nodes,
                                const std::vector<std::vector<int>>& adjacency_list,
                                double time_limit_sec = 29.3,
                                uint32_t seed = 0) {
    if (number_of_nodes <= 0) return {};
    if (number_of_nodes > 512) {
        throw std::runtime_error("fastclique_run supports n<=512 in this build.");
    }
    if ((int)adjacency_list.size() < number_of_nodes) {
        // allow shorter adjacency_list but treat missing as empty
        std::vector<std::vector<int>> padded = adjacency_list;
        padded.resize(number_of_nodes);
        FastCliqueSolver solver(number_of_nodes, padded, seed);
        return solver.solve(time_limit_sec);
    }
    FastCliqueSolver solver(number_of_nodes, adjacency_list, seed);
    return solver.solve(time_limit_sec);
}


PYBIND11_MODULE(fastclique, m) {
    m.doc() = "Fast local-search maximum clique solver (n<=512)";
    m.def("run", &fastclique_run,
          py::arg("number_of_nodes"),
          py::arg("adjacency_list"),
          py::arg("time_limit_sec") = 29.3,
          py::arg("seed") = 0);
}
// c++ -O3 -march=native -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) fastclique.cpp -o fastclique$(python3-config --extension-suffix)
