# Learned Region Prediction for Transistor-Level Gate Extraction

**A target-conditioned GNN that screens *where* a logic gate lives in a transistor-level netlist, backed by an exact VF3 subgraph-isomorphism confirmer — reproduced from the source paper on fully public benchmarks, re-engineered end to end, and stress-tested until it revealed a result the paper never reports.**

This repository reproduces the method of **Seo et al., *Target Circuit Matching in Large-Scale Netlists Using GNN-Based Region Prediction*** (arXiv:2507.19518, 2025), then pushes past reproduction into original systems engineering and an honest empirical audit of *when learned pruning actually beats a fast exact matcher — and when it does not.* Every number below is backed by a command you can run against data that ships in this repo.

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white">
<img alt="PyG" src="https://img.shields.io/badge/PyTorch_Geometric-2.3+-3C2179">
<img alt="C++20" src="https://img.shields.io/badge/C++20-OpenMP-00599C?logo=cplusplus&logoColor=white">
<img alt="CUDA" src="https://img.shields.io/badge/CUDA-sparse_ops-76B900?logo=nvidia&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## TL;DR — what is new here versus the paper

The paper is a strong positive result on a **proprietary DRAM dataset** (SK hynix *Lassen DRAMB*, not publicly available) using **VF2** as the exact backend, where VF2 is slow enough (hundreds to thousands of seconds per query) that a learned screener wins by 90–96%. This repository asks the harder, more useful question: **does the same idea survive on public, reproducible benchmarks when the exact matcher is a properly engineered, parallel VF3 rather than stock VF2?**

| Dimension | Seo et al. (paper) | This repository |
|---|---|---|
| **Dataset** | Lassen DRAMB / `Dram1gbn` — proprietary, unavailable | **ISCAS-85 / ISCAS-89** — 28 public netlists, everything reproducible |
| **Exact backend** | VF2 (stock, slow) | **VF3, custom parallel C++** — OpenMP work-stealing, NLF filter, tiered deletion, hierarchical prefix + net-group pre-passes, divide-and-conquer batching |
| **Graph schema** | 8 node types, 28 edge relations | **5 node types, 22-relation vocabulary** — specialized for pure-MOSFET netlists (no R/L/C) |
| **K-hop extraction** | GNN applied once to whole graph; per-hop embeddings sampled | **All-GPU sparse-matrix-power reachability** + vectorized induced-edge construction; **bit-exact** vs reference, profiled **275 s → 7.8 s** |
| **Headline result** | Learned pruning wins big (VF2 is the bottleneck) | **Learned pruning does *not* beat full-library VF3 at this scale — and the README proves precisely why**, identifying the crossover condition that governs when the paper's approach pays off |
| **Correctness audit** | Not addressed | **Three documented failure modes** of naive per-gate screening, including a transistor **double-counting** proof (VF3 alone, no ML) |

**In one line:** the paper shows learned pruning helps *when exact matching is the bottleneck*. This repo engineers a matcher fast enough to remove that bottleneck, then measures the regime boundary the paper leaves implicit — and reports the negative result plainly instead of burying it.

---

## Why this is worth a reviewer's / recruiter's time

- **Reproduction of a 2025 paper on data the paper does not release** — required rebuilding the graph schema, the negative-sample taxonomy, the training loop, and the entire exact-matching backend from the method description alone.
- **A real HPC/ML-systems contribution:** re-casting per-node K-hop neighborhood extraction as **sparse matrix powers on the GPU**, eliminating Python graph traversal entirely, verified **bit-exact** against the reference implementation (`max |Δ| = 0.00e+00`). This is the kind of profiling-driven rewrite (275 s → 7.8 s, five documented stages) that shows up in production ML infra work.
- **Scientific honesty as a feature, not an afterthought.** The project reaches a *negative* result on ISCAS-scale netlists, states it as the headline, and explains the mechanism — rather than cherry-picking a flattering baseline. It also identifies the exact experiment that would settle the open question (does the crossover exist at 100k+ gates?).
- **Correctness reasoning under adversarial framing:** a from-first-principles demonstration that splitting a library into per-gate searches silently produces *overlapping, contradictory* transistor claims — a subtle EDA bug that a less careful pipeline would ship as a "result."

---

## Tech stack

| Layer | Tools |
|---|---|
| **Modeling** | PyTorch 2.x, PyTorch Geometric (`RGCNConv` with basis decomposition, `global_add_pool`) |
| **GPU acceleration** | `torch.sparse` COO tensors, sparse `mm` matrix powers, vectorized `gather` + `searchsorted`, region-chunked edge construction to bound VRAM |
| **Exact matcher** | Custom **C++20** VF3 engine — OpenMP parallel work-stealing, CSR graph with bitset-accelerated `has_edge`, undo-stack backtracking (zero inner-loop heap allocation), NLF neighborhood-label filter, symmetry breaking, recursion budgeting |
| **Data / tooling** | SPICE `.sp` parser (hierarchy-aware, PDK-suffix MOSFET classification, unit scaling), NetworkX, NumPy, pandas, PyYAML, tqdm |
| **Benchmarks** | ISCAS-85 (combinational) + ISCAS-89 (sequential), TSMC-style standard-cell libraries (`*D*BWP` naming) |

---

## The problem in one paragraph

A transistor-level SPICE netlist is a flat sea of MOSFETs and wires. **Gate extraction** recovers the logic design from it: proving that *these twelve transistors* form an `AOI33D0BWP`, *those four* an inverter, until every transistor is accounted for. Formally this is **subgraph isomorphism** — NP-complete in general — and the netlists are large (`S38417` has **30,656 transistors**). The classical answer is an exact matcher (VF3). The modern proposal, from the source paper, is to put a **learned screener** in front of it so the matcher only searches likely regions. This repo builds both and measures whether the second actually wins.

---

## Two operating modes

| | **Full extraction** (`inference.py`) | **Targeted search** (`find_gate.py`) — the contribution |
|---|---|---|
| Question | "Find *every* gate in the netlist." | "Find every instance of *this one* gate." |
| Method | VF3 matches the whole library at once; the GNN annotates each hit with a confidence score. | The GNN screens the K-hop region around every transistor against one target gate; VF3 exactly confirms only the flagged regions. |
| Correctness | 100% coverage, clean partition | 100% recall & precision on the target (VF3 has the final word) |
| Speed (S38417) | **0.62 s** | ~7.8 s (GPU) for one gate |

The screener may be wrong; **the output cannot be** — VF3 confirms every candidate. The GNN buys speed, never correctness.

---

## Quick start

```bash
pip install -r requirements.txt      # torch, torch-geometric, networkx, PyYAML, ...
chmod +x vf3_cpp/build/prog          # the VF3 binary needs the execute bit
```

**Find every instance of one rare gate (the regime the method is for):**

```bash
python src/find_gate.py --circuit S38417 --gate XNR4D0BWP --device cuda --gpu-extract --baseline
```

```
  Search space  : 983/30656 transistors kept  (3.2% of chip)
  GNN seeds     : 288 regions flagged  (K=4)
  Instances     : 2 found  ->  S38417_XNR4D0BWP.v
  GNN screen (cuda)                : 7.7842 s
  VF3 confirm (carved sub-circuit) : 0.0041 s
  Recall (vs full VF3): 2/2 (100%)   Precision: 2/2 (100%)
```

**Extract the complete gate-level netlist:**

```bash
python src/inference.py --circuit C432        # -> outputs/C432.v
```

---

## How the pipeline works

**1. Netlist → heterogeneous graph (`parser.py`).** Nodes are transistors *and* nets. Node types: `VDD`, `GND`, `signal`, `PMOS`, `NMOS` (5-channel one-hot). Edges encode *which terminal* connects to which net — gate/source/drain/bulk — so the graph preserves electrical role, not mere adjacency. Signal nets additionally carry reverse edges. The relation vocabulary is **22 indices** (forward types 0–7 for PMOS/NMOS × drain/gate/source/bulk, reverse types 14–21 for signal nets); this typed encoding is what lets the model tell an `AOI` from an `OAI` built from the same transistor count. *This is a deliberate simplification of the paper's 8-node / 28-edge schema, exploiting the fact that ISCAS netlists are pure-MOSFET (no capacitors, resistors, or inductors).*

**2. Ground truth (`extractor.py`).** VF3 runs the full cell library and produces the true gate-level netlist per circuit (cached in `data/vf3_out/`). Each match is a positive training region.

**3. Negatives** — faithful to the paper's four-way taxonomy: `random` (arbitrary regions), `partial` (part of a gate), `mutation` (an edge perturbed), `others` (a different gate type). The `partial` and `others` hard negatives force the model to distinguish a real `AOI22` from a fragment of an `AOI32`.

**4. Model (`model.py`) — `CircuitFilterGNN`, a target-conditioned R-GCN.** Two `RGCNConv` layers, 128 hidden channels, 22 relations, 8 bases (basis decomposition keeps parameters bounded across relations); **~221k parameters**. The candidate region and the target gate are embedded by the **same** encoder, concatenated, and scored by an MLP → `P(region contains target gate)`. Conditioning on the target means one model handles *any* gate in the library — the gate's own topology is an input, not a class label. This matches the paper's `[h_K-hop ; h_target] → MLP` formulation.

**5. Inference — two modes** as above. `find_gate.py` screens on GPU, expands each seed to its whole cell through signal nets (not power rails), carves a reduced `.sp`, and hands it to VF3 for a guaranteed-correct confirm.

---

## The investigation: does learned pruning pay off here?

The hypothesis — *screen first, match less, go faster* — had to be dismantled three times to test it honestly. All numbers below are on `S38417` (30,656 transistors) and `C432` (532 transistors), VF3 measured with the binary in this repo, GNN screen on an RTX 4050 Laptop GPU.

**The bar to beat.** Full-library VF3 on the whole chip: **0.62 s, 4,805 gates, 100% coverage.** VF3 is fast because it matches the whole library *together* with **tiered deletion** — it finds large gates first, deletes those transistors, and the remaining graph shrinks before smaller gates are searched. That mechanism is the reason for everything below.

**Wall 1 — whole-chip screening has nothing to prune.** 100% of transistors already belong to some gate. Ask "where are the gates?" and the honest answer is "everywhere": whole-library screening on `C432` keeps **532/532 transistors — 0% pruned**. (It is asking *where are the LEGO bricks?* in a model built entirely of bricks.)

**The pivot.** The only prunable framing is **"where is *this specific* gate?"** — because most of a chip genuinely isn't any particular rare gate. That is also exactly the task the paper addresses.

**Wall 2 — splitting the library double-counts transistors (VF3 alone, no ML).** Run the full library once vs. each cell type separately and union the results (`C432`):

| | Gates found | Precision | Transistor claims |
|---|---|---|---|
| Full library, together | **91** | 100% | 532 / 532 = **1.00×** (clean partition) |
| Gate-by-gate, combined | **140** | **65%** | 780 / 532 = **1.47×** (overlaps!) |

A *piece* of a large gate looks exactly like a small gate. VF3 prevents this by deleting matched transistors; split the library and that safeguard vanishes, yielding overlapping, contradictory claims. **The gate-by-gate answer is wrong before speed even enters the discussion.**

**Wall 3 — splitting the library is dramatically slower.** Tiered deletion is also the *source* of VF3's speed. One cell type against the full chip (`S38417`): a single inverter takes **105.7 s**, one `AOI211D0BWP` 22.7 s, one `XNR4D0BWP` 20.3 s. Across ~55 cell types that extrapolates to **≈35 minutes** vs **0.62 s** run together — a **~3,000× penalty** purely for splitting the work.

**What does work: single rare-gate search.** `S38417`, target `XNR4D0BWP` (2 instances in 30,656 transistors): **3.2% of the chip kept, 100% recall, 100% precision**, VF3 confirm on the carved circuit **0.004 s**. The pruning mechanism is real.

**The verdict.** The 1.2× win reported by naive timing is a win over the *wrong opponent* — crippled single-cell VF3. The configuration a real user runs is **full-library VF3: 0.62 s for every gate**, from which XNR4 instances can simply be filtered. That is **~12× faster than the learned method and returns strictly more information.**

> **Root cause, stated plainly:** the screener has a **fixed cost** — it must look at the whole chip once, ~8 s (≈2 s with the optimization noted below). On these netlists that fixed cost is *larger than the entire matching problem it is trying to accelerate* (0.62 s). It is a toll booth that costs more than the drive.

**Learned pruning pays only when the exact matcher is itself the bottleneck** — i.e., when matching takes seconds to minutes, as VF2 does in the paper's DRAM setting. **On ISCAS-scale netlists with an engineered VF3, that regime does not exist.** This is the negative result, stated up front because it is the honest conclusion the evidence supports — and it is fully consistent with, not contradictory to, the paper: the paper simply operated on the other side of the crossover.

---

## Engineering deep-dive: making the screener fast

The screen must build and encode a K-hop region around **every one of 30,656 transistors**. Profiling drove five rewrites:

| Implementation | Screen time (S38417) | Bottleneck |
|---|---|---|
| `k_hop_subgraph` per transistor, threaded | **275 s** (CPU) | each call rescans all 195k edges for a ~21-node result |
| Neighbour-list BFS, per-region tensors | 62 s | 30k separate `torch.tensor()` constructions |
| Pure-Python BFS, batched tensors | 80 s | the Python graph-walk itself; GPU idle |
| Multiprocess BFS | sub-linear | racing parallel C++ VF3 on the CPU is a losing game |
| **Sparse-tensor extraction on GPU** (`--gpu-extract`) | **7.8 s** | no Python graph-walk at all |

The final approach eliminates CPU traversal entirely:

1. **Reachability by sparse matrix powers.** Build adjacency `A` and an `[N × C]` centre indicator as sparse tensors; `torch.sparse.mm(A, reach)` iterated K times yields *all* K-hop neighborhoods simultaneously on the GPU (**0.42 s** for all 30,656 regions).
2. **Induced edges by vectorized gather + `searchsorted`.** For every node in every region, gather out-edges via a CSR pointer, keep only those whose target lies in the *same* region — a binary search over sorted `(region, node)` keys. No loops.
3. **Chunking around the power rails.** `VDD`/`GND` appear in nearly every region with out-degree ≈25,000; expanding their edges globally allocates **12.4 GB** and OOMs. Chunking edge construction by region bounds the intermediate.

**Every rewrite was verified bit-exact** against reference `k_hop_subgraph` — `max |Δ| = 0.00e+00` on all embeddings at K = 2, 3, 4. Speed was never bought with silent approximation. *Residual, honestly flagged:* because the rails touch everything, `edges_encode` still generates and discards excess candidate edges; special-casing `VDD`/`GND` should cut the screen to **≈2 s** (not yet implemented).

---

## Model and training

| | |
|---|---|
| Architecture | Target-conditioned R-GCN, 2 layers, 128 hidden, 22 relations, 8 bases |
| Parameters | ~221,409 |
| Training set | 25 circuits, 94 gate types → 11,107 positives + 27,832 negatives |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4, grad-clip 1.0 |
| Schedule | ReduceLROnPlateau on val-F1; early stopping (patience 60) |
| Best checkpoint | epoch 285 |

**Validation:** Recall **0.955**, Precision 0.677, F1 0.792, Accuracy 0.855.

**Recall is the metric that matters** and is deliberately optimized: a missed region is a gate lost forever (VF3 never looks there), while false positives only cost a little confirm time and are eliminated downstream. **0.955 recall with VF3 as backstop is the right operating point**, which is why end-to-end targeted-search recall is 100%. The moderate precision reflects genuinely hard negatives (a fragment of a larger gate, or a structurally similar different gate, across 94 cell types) — not directly comparable to the paper's figures, which use easier randomly-sampled negatives on a different dataset.

---

## Repository layout

```
├── src/
│   ├── parser.py       SPICE (.sp) → heterogeneous graph JSON
│   ├── extractor.py    VF3 ground truth + K-hop dataset construction
│   ├── model.py        CircuitFilterGNN — target-conditioned R-GCN
│   ├── data_loader.py  Dataset / splits
│   ├── train.py        Training loop (early stopping, LR plateau)
│   ├── inference.py    FULL EXTRACTION: VF3 finds all gates, GNN annotates
│   └── find_gate.py    TARGETED SEARCH: GNN screens, VF3 confirms  ← the contribution
├── vf3_cpp/            Custom parallel VF3 exact matcher (C++20 + OpenMP) + cell libraries
├── data/
│   ├── raw/            28 ISCAS-85/89 SPICE netlists (C17 … S38417)
│   ├── parsed/         Cached graph JSON
│   └── vf3_out/        Ground-truth gate-level Verilog from full-library VF3
├── checkpoints/        Trained model (best_model.pt)
├── configs/config.yaml All hyperparameters
└── outputs/            Generated Verilog
```

---

## Reproducing every number

```bash
chmod +x vf3_cpp/build/prog

# The bar: full-library VF3, whole chip (0.62 s, 4805 gates, 100% coverage)
./vf3_cpp/build/prog -l vf3_cpp/examples/lib/libs38417.sp -s data/raw/S38417.sp -o /tmp/full.v
grep -E "Runtime|Instances|Coverage" /tmp/full.v

# What works: single rare-gate targeted search
python src/find_gate.py --circuit S38417 --gate XNR4D0BWP --device cuda --gpu-extract --baseline

# Full extraction (the practical path)
python src/inference.py --circuit S38417
```

**Provenance.** VF3 timings measured directly with the binary in this repo. GNN screen timings measured on an RTX 4050 Laptop GPU. The hierarchical-matcher figure (0.024 s) is quoted from a collaborator's benchmark and not independently reproduced here.

---

## Limitations and honest caveats

- **The method does not beat full-library VF3 at this scale.** This is the headline finding, not a footnote.
- **Timings are hardware-dependent**, but the *ratio* — screener fixed cost ≫ total matching cost — is structural, not an artefact.
- **The screener helps only for rare targets.** Common gates (e.g. inverters, 4,585 instances) prune poorly and suffer low single-cell precision.
- **Precision without VF3 would be unacceptable** (~0.68). Output exactness depends entirely on VF3 confirming every candidate — by design.
- **The `.v` output is a match list**, not fully elaborated structural Verilog with pin-to-net bindings.
- **Untested above 30k transistors.** Every conclusion is scoped to ISCAS-85/89.

---

## Future work — the one decisive experiment

Learned pruning wins only when exact matching is slow, which happens only on much larger designs. So: scale to **hundreds of thousands to millions of gates** (EPFL arithmetic benchmarks, ITC'99 `b18`/`b19`, or an industrial netlist — first technology-mapped and expanded to transistor level), benchmark GNN-screen + VF3 against the strongest matchers across many target gates, and **characterise the crossover**: at what netlist size and target rarity does the screener's fixed cost finally become worth paying? That curve is the real scientific contribution — and if it shows the crossover is unreachable at practical scales, that is an equally valuable answer. Secondary: the `VDD`/`GND` special-case (≈8 s → ≈2 s), fully structural Verilog output, and a τ-sweep of the recall/pruning trade-off.

---

## References and license

**Method reproduced:** Seo, Seo, Lee, Kim, Shin, Sung, Park. *Target Circuit Matching in Large-Scale Netlists Using GNN-Based Region Prediction.* arXiv:2507.19518, 2025.

**Exact matcher:** Carletti, Foggia, Saggese, Vento. *Challenging the Time Complexity of Exact Subgraph Isomorphism for Huge and Dense Graphs with VF3.* IEEE TPAMI 40(4), 2018. (Parallel C++ implementation in `vf3_cpp/`.)

**Library ordering:** Rajarathnam, Lin, Jin, Pan. *ReGDS: A Reverse Engineering Framework from GDSII to Gate-level Netlist.* HOST 2020.

**Benchmarks:** ISCAS-85 and ISCAS-89.

Released under the MIT License. © 2026 usuallyarnav.
