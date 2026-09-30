# Leo v5 upgrade plan

Written 2026-09-29, after leo-1.7b-v3 (released, `main`), leo-1.7b-v4 (experimental) and the naturalcodz
drop-in test. Every issue below was measured in this repo; the file or run is named next to it.

**Goal:** a Leo that matches or beats Jev on short everyday inputs (single words, fragments, one-line messages),
on classification, moderation, routing, scoring and browser agents, while staying free, local and faster. It
should also close as much of the reasoning and knowledge gap as the compute allows.

**Honest ceiling.** Beating Jev *in every case* is not reachable with a 4B base. Jev scores about 0.86 on
translated MMLU, and that knowledge comes from pretraining at a scale we cannot fine-tune into a small model.
The plan is therefore tiered:

- **Leo-4B** beats or matches Jev on everything that depends on reading the input (short conditions,
  classification, moderation, routing, browser). It stays behind on world knowledge.
- **Leo-30B-A3B** (a mixture-of-experts base with about 3B active parameters, rented GPU) is the only realistic
  way to also compete on knowledge and hard reasoning.

---

## 1. Where we stand

| benchmark | leo-1.7b-v3 | leo-1.7b-v4 | Jev 1.13 |
|---|---|---|---|
| naturalcodz drop-in, 37 scenarios | 32/37 | not run | 36/37 |
| Browser, jev-ultrafast (21 runs) | 18/21 | 12/21 | 18/21 |
| JevBench all / standard / hard | 0.697 / 0.958 / 0.396 | 0.680 / 0.903 / 0.396 | 0.861 / 0.986 / 0.721 |
| Held-out classification, mean accuracy | 0.628 | 0.627 | 0.689 |
| Multilingual (Belebele / MMMLU / INCLUDE) | 0.692 / 0.479 / 0.491 | 0.680 / 0.471 / 0.503 | 0.917 / 0.861 / 0.775 |
| Option-order flip rate (emotion / fin_topic) | 0.100 / 0.217 | 0.111 / 0.188 | 0.018 / 0.069 |
| Latency, 1 / 50 questions | 37 / 392 ms (GPU only) | same | 56 / 78 ms server, ~330 ms with network |

---

## 2. Every known issue

### A. Short inputs and phrasing (naturalcodz test, `D:\Leo-API\naturalcodz\leo-compare`)

| # | issue | evidence | cause |
|---|---|---|---|
| A1 | Bare conditions get "yes" | `is(calm text, "asking for a refund")` gives P(yes) 0.64; phrased as "Is this message asking for a refund?" it gives 0.00. `is(calm, "angry")` is true | Almost every `noul` in training is a full question or statement. Fragments like "angry" were never seen, so the model falls back to a prior near 0.6 |
| A2 | Multi-rule guards over-trigger | `contentFilter` on a prompt-injection line: PII 0.74, hate speech 0.55 (Jev 0.28 / 0.01) | No moderation data with many rules per text, and no hard negatives (texts that are bad in one way but not in the others) |
| A3 | Bare-label `pick` confuses near neighbours | "quote for the enterprise plan" went to `billing`, not `sales` | Option keys without descriptions are rare in training for business labels |
| A4 | `rate` and `score` lean toward the middle | "typo on About page" rated `medium` (Jev `low`); `score` dev accuracy 0.65 | Softmax head over levels, little ordinal data, and RPS only as a side term |
| A5 | Very short states (one word, an emoji, "ok", "refund?") are untested | no benchmark covers them | Training states average several sentences |

### B. Accuracy gaps (JevBench, held-out and multilingual; `results/`)

| # | issue | evidence |
|---|---|---|
| B1 | Hard reasoning | JevBench hard 0.396 vs 0.721. Per family: ambiguous 0.14, tradeoff 0.17, long_policy 0.21, temporal_numeric 0.20, adversarial 0.33, multi_hop 0.44 |
| B2 | World knowledge | MMMLU 0.479, INCLUDE 0.491 vs Jev 0.86 / 0.78 |
| B3 | Low-resource languages | Belebele Yoruba 0.20, Swahili 0.44, Bengali 0.64, Hindi 0.62 |
| B4 | Many-label topic tasks | fin_topic (20 labels) 0.456 vs Jev 0.669 |
| B5 | Catch-all labels under-used | daily_dialog macro-F1 0.33, "no emotion" and "other" picked less often than right |
| B6 | v4 regressions | more "yes" on policy and trap items, less "other"/"unknown" (JevBench standard 0.958 to 0.903) |

### C. Robustness and calibration

| # | issue | evidence |
|---|---|---|
| C1 | Option-order sensitivity | top-answer flip rate 5 to 6 times Jev's |
| C2 | Calibration away from dev | dev ECE 0.014; held-out ECE 0.05 to 0.17 |
| C3 | Cross-question drift in bf16 | up to 0.02 when another question is added (fp32 is exact) |
| C4 | Long states silently cut | 3 MB state accepted, cut to about 2k tokens, answered 0.49 with no warning. Trained on 2,048-token states only |

### D. Browser agent (`results/browser/`)

| # | issue | evidence |
|---|---|---|
| D1 | v3: false DONE | Google Flights: DONE with P 0.985 on a seat-class list, 0.999 on a blank render |
| D2 | v4: search bias | never says DONE on an item reached without a search: 0/6 on the reading-room tasks, all hitting the 60-action budget |
| D3 | Google Flights never passes | both versions 0/3 (Jev 0/3 and 1/3 across sessions) |
| D4 | Slow per decision | 10 to 48 s per live task vs Jev's 1.5 to 13 s, mostly because Leo takes more steps |

### E. Serving and speed

| # | issue | evidence |
|---|---|---|
| E1 | Latency grows with question count | 50 questions: 392 ms (Jev server 78 ms). Every question is re-packed with the state |
| E2 | Long state is slow | ~3.5k-token state: 1.2 s on the RTX 3050 |
| E3 | Single GPU, no batching across requests | 22 req/s under 8 clients |

### F. Process

| # | issue |
|---|---|
| F1 | The simulator overstates progress: v4 scored 100% on the held-out probe but dropped from 18/21 to 12/21 on real tasks |
| F2 | No release gate: v4 shipped regressions that were only found after training |
| F3 | JevBench was read during data design, so it is not blind; no blind short-input suite exists |
| F4 | Outside Leo: naturalcodz ignores `configure({ baseUrl })` (it passes `baseUrl`, the SDK reads `baseURL`) |

---

## 3. Root causes

1. **Training phrasing is too narrow** (A1, A3, A5, B6). Questions are well formed and descriptive. Real
   callers such as naturalcodz send fragments, bare labels and one-word states.
2. **The simulator teaches shortcuts** (D1, D2, F1). Whatever a simulated family never varies becomes a rule.
   Examples: "results always visible" in v3; "every goal starts with a search" in v4.
3. **Head design** (A4, B5, C1). A pointer head with softmax treats score levels as unordered. It has no explicit
   "none of these" prior and no mechanism that makes it ignore option order.
4. **Base size** (B1, B2, B3). Knowledge and multi-step reasoning track base-model scale more than anything
   fine-tuning adds.
5. **Serving layout** (E1, E2, C4). State tokens are recomputed per packed row, and truncation is silent.

---

## 4. Architecture v5

Leo already runs on any Hugging Face causal decoder whose layers are all softmax attention (`leo/model.py`,
`supports_block_mask`). The base is a config value, not a commitment to Qwen. Training a base model from scratch
is out of reach, since the knowledge in every strong base comes from trillions of pretraining tokens. What we can
own is everything around the base. Changes, in order of expected value:

### 4.1 Base model choice (decide in Phase 1, re-check what is current first)

| candidate | licence | why | concern |
|---|---|---|---|
| Qwen3-4B-Base (already downloaded) | Apache-2.0 | same tokenizer and code path as v3; strong multilingual | same family, same blind spots |
| Phi-4-mini (3.8B) | MIT | strong reasoning for its size | weaker multilingual |
| IBM Granite 3.x 8B | Apache-2.0 | enterprise-style classification data in pretraining | 8B is slow on T4 |
| OLMo 2 7B | Apache-2.0 | fully open training data (clean licence story) | weaker multilingual |
| Qwen3-30B-A3B (MoE) | Apache-2.0 | about 3B active parameters, so fast; knowledge close to a 30B dense model | needs one 80 GB GPU to train (rented) |

Gemma and Llama were left out because their custom licences add usage restrictions to an Apache-2.0 project.
Hybrid linear-attention bases (Qwen3.5 / Qwen3-Next) work, but they force one question per row, which is slower.

**Recommendation:** run a 3-way pilot (Qwen3-4B, Phi-4-mini, Granite 8B) on 10% of the v5 data with identical
settings, and pick by the blind short-input suite plus JevBench hard. A 30B-A3B run follows only if the budget is
approved.

### 4.2 Condition canonicaliser (fixes A1 without training; ship first)

A deterministic step in `leo/render.py`. When a `noul` instruction is a fragment (no question mark, no verb in
the second person, a few words), it is rendered with a fixed learned template, `Is the following true of the
input: <condition>?`, plus a marker. The original text stays visible to the model. It is verified by the
naturalcodz suite before any retraining. Training data (§5.1) then teaches fragments directly, and the template
stays as a safety net.

### 4.3 Order-invariant option reader (C1, B4)

- Add a 2-layer **set-attention block** over the option end states inside the head. Each option sees the others
  with no position information, so the readout does not depend on order.
- Add an **order-consistency loss**: two random option orders per question in the same batch, with the symmetric
  KL between their distributions (weight λ, start 0.5).
- Target: flip rate at or below Jev's. Keep `--order-views` as a fallback.

### 4.4 Ordinal head for `score` (A4)

Replace the softmax over levels with a **cumulative-link (CORAL-style) head**: K−1 ordered thresholds on one
latent score. It returns the same `probabilities` and `score` fields, so the wire format does not change. Ordinal
data (§5.4) and full RPS weight go with it.

### 4.5 Explicit "none" prior (B5, B6)

A learned bias logit added to any option whose key or description matches the none/other/unknown family. It is
trained on data where such options are right about 30% of the time they appear (as v3 does today), and
calibrated separately.

### 4.6 Pause tokens for reasoning (B1)

Insert K learned scratch markers (start with K = 8) between the question and `<decide>`. It is still one
forward pass and nothing is generated, but the model gets extra computation per question before answering (the
idea of "pause tokens", Goyal et al. 2023). It costs K tokens per question. It is enabled per question type and
ablated in the pilot: kept only if JevBench hard improves by 3 or more points.

### 4.7 Open-teacher soft labels (B1, B2)

For reasoning and knowledge families only: soft label distributions from an open-weight reasoning model run
locally or on a rented GPU (for example a Qwen3-32B-class model in thinking mode). They are mixed with gold labels
wherever gold exists. The allowed teachers are open-licence models; **never Jev** (TypeSafe MCA §2.3).

### 4.8 Long states (C4, E2)

- Train on states up to 8k tokens (20% of batches), with rope scaling if the base needs it.
- **No silent truncation:** the server either returns 422 with the token count, or cuts with head+tail and adds
  `usage.truncated: true`, per a server flag. The default is to reject.
- Optional **retrieval pre-pass** for states over 16k tokens: split into chunks, score chunk relevance with the
  same model, and keep the top chunks. Off by default.

### 4.9 Serving speed (E1, E3)

- **State KV reuse:** encode the state once, cache its keys and values, then run all questions as one batch
  against the cached prefix. 50 questions should cost about state + 50 short suffixes, not 50 re-packs.
- **Dynamic batching across requests** in the server (collect for up to 5 ms, then run).
- **Optional int8 / fp8 weights** for the backbone at serve time. Each is kept only if the quality gate passes.
- Target on the RTX 3050 with a 4B base: 1 question at or under 60 ms, 50 questions at or under 150 ms.

### 4.10 Calibration upgrades (C2, C3)

- Vector scaling per type and bucket (temperature plus a per-slot bias for yes/no and none options).
- **`POST /v1/calibrate`**: the user uploads a few hundred labelled requests, and the server fits and stores
  per-tenant temperatures. This is the only realistic fix for domain shift.
- Serve fp32 attention accumulation for the readout rows, to remove the 0.02 bf16 drift.

### 4.11 LoRA capacity

Move from r16 to r64 on a 4B base (about 80M trained parameters), or full fine-tuning of the top 25% of layers
if a rented GPU allows. Decided by the pilot.

---

## 5. Data plan v5

Rules carried forward: no Jev outputs, ever; labels come from datasets, code or open-licence teachers;
licence of every source recorded in the manifest; blind benchmarks never trained on.

### 5.1 Short-condition family (new, highest priority: A1, A5)

For every labelled attribute in the existing sources (intent, topic, sentiment, toxicity, spam, urgency,
policy), generate `noul` questions in **all** of these surface forms, true and false balanced 50/50:

| form | example |
|---|---|
| single word | `angry`, `spam`, `urgent`, `refund` |
| adjective phrase | `very angry`, `slightly annoyed` |
| gerund phrase | `asking for a refund`, `complaining about delivery` |
| noun phrase | `refund request`, `a billing question` |
| statement | `The customer wants a refund.` |
| question | `Is the customer asking for a refund?` |
| imperative | `check if the user is angry` |
| negated | `not spam`, `isn't about pricing` |
| with typos and case noise | `angyr`, `REFUND`, `asking 4 refund` |

- **Hard negatives** are the core of it: calm text + "angry"; a thank-you note + "asking for a refund"; a refund
  *mentioned* but not requested ("I don't need a refund, just a replacement").
- **Tiny states:** one word, an emoji, "ok", "refund?", a URL, a number, and empty-ish text, with correct answers
  (usually "no" or "not stated").
- Target size: 60k questions. The same generator also builds `choice` with **bare label keys** and near-synonym
  distractors (sales vs billing vs account, bug vs feature request vs question) for A3.

### 5.2 Moderation with many rules (A2)

- Sources (licences checked at build time): Jigsaw civil_comments, ToxiGen, deepset prompt-injections and
  jackhhao jailbreak (already in use), an open PII corpus such as ai4privacy (check the licence), plus a
  **synthetic PII generator** with fake emails, phones, IDs and addresses, and **look-alike negatives** (code
  snippets, order numbers, public business addresses, injection text with no PII).
- Every text is asked **all six** naturalcodz `contentFilter` rules at once, so the model learns that one bad
  property does not imply the others.

### 5.3 Reasoning (B1)

| JevBench family | data | label source |
|---|---|---|
| long_policy, policy exceptions | generated policies of 1k to 6k tokens with nested exceptions; ContractNLI | code for generated; dataset for ContractNLI |
| temporal_numeric | dates, durations, arithmetic in prose; DROP (as choice) | code; dataset |
| multi_hop | tables + records with 2 to 4 hops; HotpotQA / MuSiQue as choice | code; dataset |
| ambiguous, tradeoff | scenarios with explicit criteria weights | open teacher + code consistency check, 5% human spot check |
| adversarial, trap | instructions hidden in state, contradictory criteria | code |

### 5.4 Ordinal and scoring (A4)

Yelp and Amazon-style stars (already in use), HelpSteer2 (in use), plus generated severity and urgency scales
where the level is computed from explicit facts (downtime, users affected, workaround). Every scale is also
emitted in the naturalcodz `rate(…, 1, 5)` form ("1 out of 5" … "5 out of 5").

### 5.5 Multilingual and low-resource (B3)

AfriSenti, MasakhaNEWS, NusaX, IndicXNLI and IndicSentiment, plus the §5.1 short-condition family machine-translated
with NLLB (an open model) into 20 languages. Belebele, MMMLU and INCLUDE stay blind.

### 5.6 Knowledge (B2)

ARC, OpenBookQA, SciQ, CommonsenseQA (in use), the MMLU auxiliary train split, and TriviaQA reformatted as
choice with retrieved distractors. Expect small gains on a 4B base; the real lever is the 30B-A3B base.

### 5.7 Browser (D1 to D4)

- Keep the v4 evidence family, and **remove its shortcut**. 40% of goals open an item **without** searching
  (article lists, menus, direct links). DONE is right the moment the requested item is open.
- New simulated site types: reading rooms and blogs, multi-step forms, date pickers, paginated lists, and
  cookie or consent overlays.
- More Mind2Web steps (all train splits), plus **self-play traces**: Leo runs on held-in live sites, and each step
  is labelled by the independent page checks and a scripted oracle, never by Jev.
- The browser benchmark sites and goals stay held out.

### 5.8 Order augmentation and replay

Every choice question appears in two orders per epoch (for the §4.3 loss). The full v3 mixture is replayed, so
nothing is forgotten; the v4 lesson is not to train a narrow family on its own.

**Expected size:** about 250k requests (v3 had 84k). Language mix: English 55%, other languages 45%.

---

## 6. Evaluation and release gate

New blind suites, built before training and never read while designing data:

1. **Short-input suite:** 600 items across all §5.1 forms and 10 languages, hand-checked.
2. **naturalcodz suite:** the 37 scenarios plus 300 new ones, run through the real library (both backends).
3. **Browser suite v2:** the current 7 tasks plus 8 new ones (reading, forms, date pickers, consent overlays),
   5 repeats each.

Also every existing benchmark: JevBench, held-out four, multilingual, probes, false-DONE probe, latency.

**A version ships to `main` only if** all of the following hold:

- no existing benchmark drops by more than 1 point;
- the browser pass count is at least the current release's;
- the short-input and naturalcodz suites beat the current release;
- calibrated ECE on held-out data is at or below the current release's;
- the pilot of every architecture change showed a gain.

The comparison is generated by `scripts/export_hf.py` as today. A failed gate means the version goes to a branch.

---

## 7. Targets

| benchmark | v3 today | Leo-4B target | Leo-30B-A3B target | Jev 1.13 |
|---|---|---|---|---|
| naturalcodz suite | 32/37 | ≥ 36/37 | ≥ 37/37 | 36/37 |
| short-input suite (new) | not measured | ≥ Jev | ≥ Jev | to measure |
| browser | 18/21 | ≥ 19/21 on v1 tasks | ≥ 20/21 | 18/21 |
| JevBench hard | 0.396 | 0.55 | 0.70 | 0.721 |
| held-out mean | 0.628 | 0.68 | 0.71 | 0.689 |
| MMMLU | 0.479 | 0.60 | 0.78 | 0.861 |
| option-order flip, emotion | 0.100 | ≤ 0.02 | ≤ 0.02 | 0.018 |
| latency, 50 questions (local GPU) | 392 ms | ≤ 150 ms | ≤ 150 ms (A100) | 78 ms server |

The targets are estimates for planning, not promises. The pilot in Phase 2 replaces them with measured numbers.

---

## 8. Compute

| run | hardware | estimate |
|---|---|---|
| v3 (for reference) | 2x T4, Kaggle | 84k requests x 2 epochs = 15.3 h |
| Leo-4B pilot (3 bases x 10% data) | 2x T4, Kaggle | about 3 x 3 h |
| Leo-4B full (250k requests, 1.5 epochs) | 2x T4, Kaggle | about 55 to 70 h: more than one account's weekly quota; better on a rented GPU |
| Leo-4B full | 1x H100, rented | about 8 to 12 h, roughly $25 to $50 |
| Leo-30B-A3B full | 1x H100 80 GB, rented | about 25 to 40 h, roughly $80 to $150 |

**Kaggle caution:** Kaggle's terms allow one account per person. Running training on a second personal account
risks both accounts being banned. The second account should only be used if it belongs to a different person
who runs it themselves. Otherwise use rented GPUs for the full runs.

---

## 9. Tunable parameters

| parameter | where | now | v5 range |
|---|---|---|---|
| LoRA rank / alpha | train | 16 / 32 | 32 to 64 / 2x rank |
| learning rate, head LR | train | 2e-4, 1e-3 | 1e-4 to 3e-4, 5e-4 to 1e-3 |
| RPS weight | train | 0.5 | 1.0 with ordinal head |
| order-consistency λ | train (new) | none | 0.25 to 1.0 |
| pause tokens K | model (new) | 0 | 0, 4, 8, 16 |
| set-attention layers | head (new) | 0 | 0, 1, 2 |
| none-prior bias | head (new) | none | learned |
| train state tokens | train | 2,048 | 2,048 (80%), 8,192 (20%) |
| label smoothing | train | 0 | 0 to 0.05 (gold labels only) |
| family mix weights | data | fixed | tuned on dev per family |
| temperatures / vector scaling | calibrate | per type and bucket | plus per-slot bias, per-tenant fit |
| `--order-views` | serve | 1 | 1 (if §4.3 works) or 2 |
| truncation policy | serve (new) | silent cut | reject / head-tail with flag |
| decision thresholds | client | naturalcodz defaults | re-tuned per backend from calibration data |

---

## 10. Execution steps

Each phase ends with a check-in, and nothing large starts without sign-off.

**Phase 0: hotfixes on v3, no training (1 day)**
1. Condition canonicaliser (§4.2), measured on the naturalcodz suite and the 1-word probes.
2. Server: explicit truncation policy (§4.8) and a `usage.truncated` flag.
3. naturalcodz: fix `baseUrl` → `baseURL` in `src/client.ts` and `src/createNatural.ts`.
4. Release as leo-1.7b-v3.1 if the gate holds.

**Phase 1: suites and data (4 to 6 days)**
1. Build the blind short-input, naturalcodz-300 and browser-v2 suites; score v3 and Jev on them (Jev scoring only).
2. Implement the §5 generators and converters, with tests for every family (label balance, no benchmark overlap).
3. Build `data/processed-v5` and manifest; audit shortcuts per family (the `none_audit.py` approach, extended to
   history-length, search-presence and label-position checks).

**Phase 2: architecture and pilot (4 to 5 days)**
1. Implement §4.3 to §4.6 and §4.10 behind flags, each with unit tests (order invariance of the head,
   ordinal monotonicity, KV-reuse parity with packed rows).
2. Pilot: 3 bases x 10% data, then ablate each flag on the best base. Keep only changes that win on dev and
   the blind suites.
3. **Sign-off:** base model, kept flags, compute budget.

**Phase 3: full training (1 to 3 days of GPU time)**
1. Full run on the chosen base (rented GPU recommended), with resume and checkpoint selection as today.
2. Calibration and per-slot bias fitting.

**Phase 4: evaluation and release (2 days)**
1. Every suite, both backends, generated report.
2. Release gate (§6). Pass: Hugging Face `main` + GitHub tag. Fail: branch plus written findings.
3. Serving: KV reuse and dynamic batching (§4.9) and a Docker image with a key, rate limit and HTTPS guidance.

**Phase 5 (optional, budget permitting): Leo-30B-A3B**
Same data and flags, one rented 80 GB GPU, same gate.

---

## 11. Decisions needed before Phase 1

1. **Blind-set policy.** Training on emotion data would help naturalcodz's `is(text, "angry")` but would end the
   "emotion is held out" claim. Proposal: allow emotion-family training data, and use the new blind suites as the
   headline instead.
2. **Compute:** Kaggle (one account) for the pilots, and a rented GPU (about $25 to $50) for the 4B full run.
   Optional $80 to $150 for 30B-A3B.
3. **Base shortlist** for the pilot (§4.1).
4. **Teacher model** for §4.7 soft labels, and whether to rent a GPU to run it.
