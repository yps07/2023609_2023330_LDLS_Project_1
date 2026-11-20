// gdn_bench.cpp -- Option 5: time the ggml_gated_delta_net operator (decode + prefill)
// Shapes follow Qwen3-Next: H_k=16, H_v=32, head_size=128 (v_repeat=2).
// Built against ~/llama.cpp (commit 51a77c4), CPU backend, -O3 -march=native.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

static void fill_uniform(float* p, size_t n, float mn, float mx) {
    // deterministic fill (values do not affect FLOP count / timing)
    for (size_t i = 0; i < n; i++) {
        float t = (float)((i * 2654435761u) & 0xffff) / 65535.0f;
        p[i] = mn + (mx - mn) * t;
    }
}

struct BenchResult { int n_tokens; int n_threads; double mean_us; double std_us; double min_us; };

static BenchResult run_one(int n_tokens, int n_threads, int warmup, int iters, int K, bool permuted) {
    const int64_t Hk = 16, Hv = 32, S = 128;
    const int v_repeat = Hv / Hk; // 2
    const int64_t n_seqs = 1;

    size_t ctx_mem = 256 * 1024 * 1024;
    ggml_init_params iparams = { ctx_mem, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(iparams);
    // weight/output context not needed separately

    ggml_tensor* q; ggml_tensor* k; ggml_tensor* v;
    if (permuted) {
        q = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, Hk, n_seqs), 0, 2, 1, 3);
        k = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, Hk, n_seqs), 0, 2, 1, 3);
        v = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, n_tokens, Hv, n_seqs), 0, 2, 1, 3);
    } else {
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_tokens, n_seqs);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_tokens, n_seqs);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, n_tokens, n_seqs);
    }
    ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v");

    const int64_t g_ne0 = /* kda */ false ? S : 1;
    ggml_tensor* g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g_ne0, Hv, n_tokens, n_seqs);
    ggml_tensor* beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, n_tokens, n_seqs);
    ggml_tensor* state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, n_seqs);
    ggml_set_name(g, "g"); ggml_set_name(beta, "beta"); ggml_set_name(state, "state");

    // init
    fill_uniform((float*)q->data, ggml_nelements(q), -1.0f, 1.0f);
    fill_uniform((float*)k->data, ggml_nelements(k), -1.0f, 1.0f);
    fill_uniform((float*)v->data, ggml_nelements(v), -0.3f, 5.0f);
    fill_uniform((float*)g->data, ggml_nelements(g), -20.0f, -1e-4f);
    fill_uniform((float*)beta->data, ggml_nelements(beta), 0.0f, 1.0f);
    fill_uniform((float*)state->data, ggml_nelements(state), -1.0f, 1.0f);

    // l2 norm q,k as in the model
    q = ggml_l2_norm(ctx, q, 1e-6f);
    k = ggml_l2_norm(ctx, k, 1e-6f);

    ggml_tensor* out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
    ggml_set_name(out, "out");

    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    // compute (CPU, n_threads via graph compute)
    for (int i = 0; i < warmup; i++) {
        ggml_graph_compute_with_ctx(ctx, graph, n_threads);
    }

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_graph_compute_with_ctx(ctx, graph, n_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }

    double sum = 0, mn = times[0];
    for (double x : times) { sum += x; mn = std::min(mn, x); }
    double mean = sum / times.size();
    double var = 0;
    for (double x : times) var += (x - mean) * (x - mean);
    double stdv = std::sqrt(var / times.size());

    ggml_free(ctx);

    BenchResult r; r.n_tokens = n_tokens; r.n_threads = n_threads;
    r.mean_us = mean; r.std_us = stdv; r.min_us = mn;
    return r;
}

int main(int argc, char** argv) {
    bool exp = false, dump = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--exp") exp = true;
        if (std::string(argv[i]) == "--dump") dump = true;
    }

    if (dump) {
        const int n_tokens = 1, n_threads = 4, K = 1;
        const int64_t Hk = 16, Hv = 32, S = 128;
        const int64_t n_seqs = 1;
        size_t ctx_mem = 64 * 1024 * 1024;
        ggml_init_params iparams = { ctx_mem, nullptr, false };
        ggml_context* ctx = ggml_init(iparams);
        ggml_tensor *q, *k, *v;
        q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_tokens, n_seqs);
        k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hk, n_tokens, n_seqs);
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, Hv, n_tokens, n_seqs);
        ggml_set_name(q, "q"); ggml_set_name(k, "k"); ggml_set_name(v, "v");
        ggml_tensor* g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, n_tokens, n_seqs);
        ggml_tensor* beta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, Hv, n_tokens, n_seqs);
        ggml_tensor* state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, Hv, n_seqs);
        ggml_set_name(g, "g"); ggml_set_name(beta, "beta"); ggml_set_name(state, "state");
        auto fill = [](ggml_tensor* t, float mn, float mx) {
            size_t n = ggml_nelements(t);
            float* p = (float*)t->data;
            for (size_t i = 0; i < n; i++) { float tt = (float)((i*2654435761u)&0xffff)/65535.0f; p[i] = mn + (mx-mn)*tt; }
        };
        fill(q, -1, 1); fill(k, -1, 1); fill(v, -0.3f, 5.0f);
        fill(g, -20.0f, -1e-4f); fill(beta, 0.0f, 1.0f); fill(state, -1.0f, 1.0f);
        q = ggml_l2_norm(ctx, q, 1e-6f);
        k = ggml_l2_norm(ctx, k, 1e-6f);
        ggml_tensor* out = ggml_gated_delta_net(ctx, q, k, v, g, beta, state, K);
        ggml_set_name(out, "out");
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_graph_compute_with_ctx(ctx, graph, n_threads);
        size_t nels = ggml_nelements(out);
        float* p = (float*)out->data;
        double s = 0, sq = 0; float mn = p[0], mx = p[0];
        bool finite = true;
        for (size_t i = 0; i < nels; i++) {
            if (!std::isfinite(p[i])) finite = false;
            s += p[i]; sq += (double)p[i]*p[i];
            mn = std::min(mn, p[i]); mx = std::max(mx, p[i]);
        }
        printf("dump: nels=%zu finite=%d min=%.6f max=%.6f sum=%.6f sumsq=%.6f\n",
               nels, (int)finite, mn, mx, s, sq);
        ggml_free(ctx);
        return 0;
    }

    if (!exp) {
        printf("n_tokens,n_threads,iters,mean_us,std_us,min_us\n");
        const int toks[] = {1, 64, 256, 512, 1024};
        const int thr[]  = {1, 2, 4, 8, 12};
        for (int nt : toks) {
            for (int th : thr) {
                BenchResult r = run_one(nt, th, 20, 200, 1, false);
                printf("%d,%d,200,%.2f,%.2f,%.2f\n", r.n_tokens, r.n_threads, r.mean_us, r.std_us, r.min_us);
                fflush(stdout);
            }
        }
    } else {
        // experiment: K=1 vs K=4 at decode, and permuted vs contiguous
        printf("variant,n_tokens,n_threads,iters,mean_us,std_us,min_us\n");
        struct V { const char* name; int K; bool perm; };
        V variants[] = { {"K=1_contig",1,false}, {"K=4_contig",4,false}, {"K=1_permuted",1,true} };
        for (auto& v : variants) {
            BenchResult r = run_one(1, 4, 20, 500, v.K, v.perm);
            printf("%s,1,4,500,%.2f,%.2f,%.2f\n", v.name, r.mean_us, r.std_us, r.min_us);
            fflush(stdout);
        }
    }
    return 0;
}
