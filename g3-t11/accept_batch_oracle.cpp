// G3 t11 deepseek-a: compose three real/contract components and test the
// NGRAM topology + accept convention, including the previously untested
// batch dimension (F1 divergences were 10/12 at concurrency 4).
//
//  1. REAL tree builder: sglang::ngram::fillResult from
//     python/sglang/kernels/jit/csrc/ngram_corpus/result.cpp (compiled, not ported).
//  2. Documented reconstruct contract, transcribed independently from
//     python/sglang/kernels/ops/speculative/reconstruct_tree.py docstring
//     (retrieve_index[b,tid] = b*n+tid; first child / next sibling from parents).
//  3. REAL accept kernel: verify_tree_greedy_kernel_impl extracted VERBATIM from
//     python/sglang/kernels/aot/csrc/cpu/spec.cpp:47-92 by extract_real_sources.py.
//     Only the OpenMP dispatcher at::parallel_for is shimmed to serial; the
//     function body is not modified.
//
// Independent goldens are planted greedy sequences, never produced by the code
// under test. Fault controls must be DETECTED (i.e. produce a non-golden result);
// if a control passes, the oracle reports INSTRUMENT_INVALID.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include "result.h"  // real ngram_corpus Result/Node/fillResult

// ---- minimal serial shim for the one ATen call in the real kernel body ----
namespace at {
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t, F &&fn) {
  fn(begin, end);
}
}  // namespace at

#include "generated/verify_tree_greedy_real.h"  // real kernel body, verbatim

using namespace sglang::ngram;

struct BatchCase {
  std::string name;
  std::vector<int32_t> tokens;   // BFS token sequence including root at 0
  std::vector<int32_t> parents;  // parent row per node, -1 for root
  std::vector<int32_t> target;   // target_predict per row (row 0 = argmax after root)
  std::vector<int32_t> want_commit;  // independent golden
  int32_t want_accepted = 0;
};

// Build a corpus tree: parents[i] must be < i; children keyed by token must be
// unique per parent (fillResult uses unordered_map<int32_t,int32_t>).
static Result build_real(const BatchCase &c) {
  int n = static_cast<int>(c.tokens.size());
  std::vector<Node> tree(n);
  for (int i = 1; i < n; ++i) {
    int p = c.parents[i];
    tree[p].next[c.tokens[i]] = i;
  }
  int last_token = c.tokens[0];
  return fillResult(last_token, n, tree, 0);
}

// fillResult emits rows in BFS order driven by unordered_map iteration, so the
// real row order need not equal the semantic order used to declare the case.
// Map real row -> semantic index by unique token id (all case tokens distinct).
static std::vector<int> semantic_order(const Result &r, const BatchCase &c) {
  std::vector<int> out(r.token.size(), -1);
  for (size_t i = 0; i < r.token.size(); ++i) {
    for (size_t s = 0; s < c.tokens.size(); ++s) {
      if (c.tokens[s] == r.token[i]) { out[i] = static_cast<int>(s); break; }
    }
  }
  return out;
}

static int row_of_token(const Result &r, int32_t tok) {
  for (size_t i = 0; i < r.token.size(); ++i)
    if (r.token[i] == tok) return static_cast<int>(i);
  return -1;
}

struct Links {
  std::vector<int64_t> rix, nxt, sib;
};

// Documented reconstruct contract (reconstruct_tree.py docstring), independent
// of the accept kernel under test and of the corpus builder.
static Links reconstruct_contract(const Result &r, int n, int bs, int b,
                                  int64_t seq_len, int64_t *pos_out) {
  Links L;
  L.rix.assign(n, -1);
  L.nxt.assign(n, -1);
  L.sib.assign(n, -1);
  std::vector<int> parent(n, -1);
  std::vector<int> nanc(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      if (r.mask[i * n + j]) {  // strict ancestors only: j < i
        parent[i] = j;          // largest such j
        ++nanc[i];
      }
    }
  }
  for (int k = 0; k < n; ++k) {
    for (int i = k + 1; i < n; ++i) {
      if (r.mask[i * n + k] && k < i) { L.nxt[k] = i; break; }  // first child
    }
  }
  for (int k = 0; k < n; ++k) {
    if (parent[k] < 0) { L.sib[k] = -1; continue; }
    for (int i = k + 1; i < n; ++i) {
      if (parent[i] == parent[k]) { L.sib[k] = i; break; }
    }
  }
  for (int i = 0; i < n; ++i) {
    L.rix[i] = static_cast<int64_t>(b) * n + i;  // global flat index
    if (pos_out) pos_out[i] = nanc[i] + seq_len;
  }
  return L;
}

// Run the REAL kernel over bs cases packed exactly as the engine packs them.
static int run_batch(const std::vector<BatchCase> &cases, bool scramble_batch_offset,
                     std::vector<std::vector<int32_t>> *commit_out,
                     std::vector<int32_t> *acc_out,
                     std::vector<Result> *results_out) {
  const int bs = static_cast<int>(cases.size());
  const int n = static_cast<int>(cases[0].tokens.size());
  const int steps = n;  // num_spec_step = draft_token_num in the engine
  std::vector<Result> results;
  for (auto &c : cases) results.push_back(build_real(c));

  std::vector<int64_t> candidates(bs * n, 0), target(bs * n, 0);
  std::vector<int64_t> rix(bs * n, -1), nxt(bs * n, -1), sib(bs * n, -1);
  for (int b = 0; b < bs; ++b) {
    Links L = reconstruct_contract(results[b], n, bs, b, 100, nullptr);
    std::vector<int> order = semantic_order(results[b], cases[b]);
    for (int i = 0; i < n; ++i) {
      if (order[i] < 0) { std::printf("FATAL: token not in case\n"); std::exit(2); }
      int64_t g = b * n + i;
      candidates[g] = (results[b].token[i]);
      target[g] = cases[b].target[order[i]];  // align by semantic node, not row
      // Contract: next_token / next_sibling are LOCAL node ids (the kernel uses
      // them as an index into the per-batch row); only retrieve_index is global.
      nxt[g] = L.nxt[i];
      sib[g] = L.sib[i];
      rix[g] = scramble_batch_offset ? i : L.rix[i];
    }
  }
  std::vector<int32_t> predicts(bs * n, -999), accept_index(bs * steps, -1),
      accnum(bs, -1);
  verify_tree_greedy_kernel_impl<int64_t>(
      predicts.data(), accept_index.data(), accnum.data(), candidates.data(),
      rix.data(), nxt.data(), sib.data(), target.data(), bs, steps, n);

  commit_out->assign(bs, {});
  acc_out->assign(bs, -1);
  for (int b = 0; b < bs; ++b) {
    // predict[accept_index] as the worker commits it; flat predicts [bs*n].
    for (int k = 0; k <= accnum[b]; ++k) {
      int32_t idx = accept_index[b * steps + k];
      (*commit_out)[b].push_back(predicts[idx]);
    }
    (*acc_out)[b] = accnum[b];
  }
  *results_out = results;
  return 0;
}

static void print_vec(const std::vector<int32_t> &v) {
  std::printf("[");
  for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? "," : "", v[i]);
  std::printf("]");
}

int main() {
  // ---- batch 0: chain root 999 -> 11 -> 12 -> 13, bonus 21
  BatchCase c0;
  c0.name = "chain";
  c0.tokens = {999, 11, 12, 13};
  c0.parents = {-1, 0, 1, 2};
  c0.target = {11, 12, 13, 21};
  c0.want_commit = {11, 12, 13, 21};
  c0.want_accepted = 3;

  // ---- batch 1: root 777 -> {5, 6}; 5 -> 7. Greedy takes sibling 6, bonus 41.
  BatchCase c1;
  c1.name = "branch_sibling";
  c1.tokens = {777, 5, 6, 7};
  c1.parents = {-1, 0, 0, 1};
  c1.target = {6, 77, 41, 88};
  c1.want_commit = {6, 41};
  c1.want_accepted = 1;

  int npass = 0, nfail = 0;
  std::vector<std::string> msgs;

  // ===== T1: bs=1, the chain =====
  {
    std::vector<std::vector<int32_t>> got;
    std::vector<int32_t> acc;
    std::vector<Result> res;
    run_batch({c0}, false, &got, &acc, &res);
    bool ok = got[0] == c0.want_commit && acc[0] == c0.want_accepted;
    std::printf("T1 bs=1 chain: ok=%d acc=%d commit=", ok, acc[0]);
    print_vec(got[0]);
    std::printf("\n");
    (ok ? npass : nfail)++;
  }

  // ===== T2: bs=1, branch takes sibling =====
  {
    std::vector<std::vector<int32_t>> got;
    std::vector<int32_t> acc;
    std::vector<Result> res;
    run_batch({c1}, false, &got, &acc, &res);
    bool ok = got[0] == c1.want_commit && acc[0] == c1.want_accepted;
    std::printf("T2 bs=1 branch_sibling: ok=%d acc=%d commit=", ok, acc[0]);
    print_vec(got[0]);
    std::printf("\n");
    (ok ? npass : nfail)++;
  }

  // ===== T3: bs=2 mixed, identical tree width; batch isolation =====
  {
    std::vector<std::vector<int32_t>> got;
    std::vector<int32_t> acc;
    std::vector<Result> res;
    run_batch({c0, c1}, false, &got, &acc, &res);
    bool ok = got[0] == c0.want_commit && acc[0] == c0.want_accepted &&
              got[1] == c1.want_commit && acc[1] == c1.want_accepted;
    std::printf("T3 bs=2 mixed: ok=%d acc=[%d,%d] commit0=", ok, acc[0], acc[1]);
    print_vec(got[0]);
    std::printf(" commit1=");
    print_vec(got[1]);
    std::printf(" (golden1 ");
    print_vec(c1.want_commit);
    std::printf(")\n");
    (ok ? npass : nfail)++;
  }

  // ===== T4: batch-offset fault control. retrieve_index without b*n must be
  // DETECTED (batch 1 commits batch 0's root argmax). If it is not detected the
  // instrument cannot see the bug class it claims to test -> INSTRUMENT_INVALID.
  {
    std::vector<std::vector<int32_t>> got;
    std::vector<int32_t> acc;
    std::vector<Result> res;
    run_batch({c0, c1}, true, &got, &acc, &res);
    bool batch1_wrong = got.size() == 2 && got[1] != c1.want_commit;
    std::printf("T4 control_no_batch_offset: detected=%d commit1=", batch1_wrong);
    print_vec(got[1]);
    std::printf(" (golden1 ");
    print_vec(c1.want_commit);
    std::printf(")\n");
    (batch1_wrong ? npass : nfail)++;
    if (!batch1_wrong) msgs.push_back("T4_NOT_DETECTED");
  }

  // ===== T5: mask-corruption control. A genuinely-zero ancestor bit
  // (mask[3,2] in the branch tree: node 2 is not an ancestor of node 3) is set,
  // re-parenting node 3 under node 2. The commit must change from the golden.
  {
    BatchCase c = c1;
    // semantic nodes: 1(token5), 2(token6), 3(token7); 1 and 3 are a chain.
    c.target = {5, 7, 77, 31};  // root->token5, token5->token7, bonus 31
    c.want_commit = {5, 7, 31};
    c.want_accepted = 2;
    std::vector<std::vector<int32_t>> got;
    std::vector<int32_t> acc;
    std::vector<Result> res;
    run_batch({c}, false, &got, &acc, &res);
    bool baseline_ok = got[0] == c.want_commit && acc[0] == 2;
    // Real fillResult gives mask[row(7)][row(6)] == 0 (6 is not an ancestor of
    // 7). Set it: 7 re-parents under 6 and the accepted path must shorten.
    Result rr = build_real(c);
    int r7 = row_of_token(rr, 7), r6 = row_of_token(rr, 6);
    bool bit_was_zero = r7 >= 0 && r6 >= 0 && rr.mask[r7 * 4 + r6] == 0;
    if (r7 >= 0 && r6 >= 0) rr.mask[r7 * 4 + r6] = 1;
    // Re-run accept on the corrupted mask by rebuilding links + real kernel.
    int n = 4;
    Links L;
    {
      std::vector<int> parent(n, -1);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
          if (rr.mask[i * n + j]) parent[i] = j;
      L.rix.resize(n); L.nxt.assign(n, -1); L.sib.assign(n, -1);
      for (int k = 0; k < n; ++k)
        for (int i = k + 1; i < n; ++i)
          if (rr.mask[i * n + k]) { L.nxt[k] = i; break; }
      for (int k = 0; k < n; ++k) {
        if (parent[k] < 0) continue;
        for (int i = k + 1; i < n; ++i)
          if (parent[i] == parent[k]) { L.sib[k] = i; break; }
      }
      for (int i = 0; i < n; ++i) L.rix[i] = i;
    }
    std::vector<int64_t> cand(n), tgt(n);
    {
      std::vector<int> order = semantic_order(rr, c);
      for (int i = 0; i < n; ++i) {
        cand[i] = rr.token[i];
        tgt[i] = c.target[order[i]];
      }
    }
    std::vector<int64_t> nxt(L.nxt), sib(L.sib);
    std::vector<int32_t> pr(n, -999), ai(n, -1), an(1, -1);
    verify_tree_greedy_kernel_impl<int64_t>(pr.data(), ai.data(), an.data(),
                                            cand.data(), L.rix.data(),
                                            nxt.data(), sib.data(), tgt.data(),
                                            1, n, n);
    std::vector<int32_t> corrupt_commit;
    for (int k = 0; k <= an[0]; ++k) corrupt_commit.push_back(pr[ai[k]]);
    bool detected = baseline_ok && bit_was_zero && corrupt_commit != c.want_commit;
    std::printf("T5 control_mask_corrupt: baseline_ok=%d bit_was_zero=%d detected=%d "
                "corrupt_commit=", baseline_ok, bit_was_zero, detected);
    print_vec(corrupt_commit);
    std::printf(" (golden ");
    print_vec(c.want_commit);
    std::printf(")\n");
    (detected ? npass : nfail)++;
    if (!detected) msgs.push_back("T5_NOT_DETECTED");
  }

  std::printf("{\"oracle\":\"g3-t11-accept-batch\","
              "\"components\":{\"tree\":\"result.cpp fillResult (compiled real)\","
              "\"links\":\"reconstruct_tree.py docstring contract (independent)\","
              "\"accept\":\"spec.cpp:47-92 verify_tree_greedy_kernel_impl (verbatim)\"},"
              "\"passed\":%d,\"failed\":%d,\"instrument_invalid\":%s}\n",
              npass, nfail, nfail ? "true" : "false");
  for (auto &m : msgs) std::printf("// %s\n", m.c_str());
  return nfail ? 1 : 0;
}
