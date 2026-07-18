# Learned Region Prediction for Transistor-Level Gate Extraction

**A GNN-based screener that narrows exact subgraph matching — plus an honest, measured account of when that helps and when it doesn't.**

This project reproduces and stress-tests and improvises the approach of Seo et al., *"Target Circuit Matching in Large-Scale Netlists Using GNN-Based Region Prediction"* (arXiv:2507.19518, 2025): train a graph neural network to predict *where* a target logic gate lives in a transistor-level netlist, then let an exact subgraph-isomorphism matcher (VF3) confirm only those regions.

The pipeline works end to end and is provably correct. What makes this repository worth reading, though, is the second half: **a rigorous, reproducible investigation into whether learned pruning actually beats a fast exact matcher on standard benchmark netlists — and the finding that, at this scale, it does not.** Every claim below is backed by a command you can run.

---

## Table of contents

- [The problem](#the-problem)
- [Two ways to solve it](#two-ways-to-solve-it)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [How the pipeline works](#how-the-pipeline-works)
- [The investigation: does learned pruning pay off?](#the-investigation-does-learned-pruning-pay-off)
  - [Wall 1 — Whole-chip screening has nothing to prune](#wall-1--whole-chip-screening-has-nothing-to-prune)
  - [The pivot to single-target search](#the-pivot-to-single-target-search)
  - [Wall 2 — Splitting the library double-counts transistors](#wall-2--splitting-the-library-double-counts-transistors)
  - [Wall 3 — Splitting the library is dramatically slower](#wall-3--splitting-the-library-is-dramatically-slower)
  - [What does work: single rare-gate search](#what-does-work-single-rare-gate-search)
  - [The verdict, and why](#the-verdict-and-why)
- [Engineering: making the screener fast](#engineering-making-the-screener-fast)
- [Model and training](#model-and-training)
- [Reproducing every number in this README](#reproducing-every-number-in-this-readme)
- [Limitations and honest caveats](#limitations-and-honest-caveats)
- [Future work](#future-work)
- [References and license](#references-and-license)

---

## The problem

A transistor-level SPICE netlist is a flat sea of MOSFETs and wires. **Gate extraction** is the task of recovering the logic design from it: finding that *these twelve transistors* form an `AOI33D0BWP`, *those four* form an inverter, and so on, until every transistor is accounted for.

Formally this is **subgraph isomorphism** — for each standard cell in a library, find every occurrence of that cell's transistor topology inside the circuit graph. It is NP-complete in general, and netlists are large: `S38417`, the biggest benchmark here, has **30,656 transistors**.

The classical solution is an exact matcher such as **VF3**. The modern proposal is to put a **learned screener** in front of it: a GNN predicts which regions are likely to contain the target gate, and the matcher only searches there. Less ground to cover should mean less time.

This repository implements both, and measures whether the second actually wins.

---

## Two ways to solve it

| | **Exact extraction** (`inference.py`) | **Learned targeted search** (`find_gate.py`) |
|---|---|---|
| **Question answered** | "Find *every* gate in the netlist." | "Find every instance of *this one* gate." |
| **Method** | VF3 matches the whole cell library at once; the GNN annotates each result with a confidence score. | The GNN screens every transistor neighbourhood against one target gate; VF3 exactly confirms only the flagged regions. |
| **Output** | Complete gate-level netlist | Verilog listing of that gate's instances |
| **Correctness** | 100% coverage, clean partition | 100% recall & precision on the target |
| **Speed (S38417)** | **0.62 s** | ~7.8 s (GPU) for one gate |

Both are implemented and both are correct. The performance comparison between them is the subject of the investigation below — and the answer is not the one we expected.

---

## Quick start

### Requirements

```bash
pip install -r requirements.txt      # torch, torch-geometric, networkx, PyYAML, ...
chmod +x vf3_cpp/build/prog          # the VF3 binary needs the execute bit
```

A CUDA GPU is optional but strongly recommended for `find_gate.py` (see [Engineering](#engineering-making-the-screener-fast)).

### Find every instance of one gate

```bash
python src/find_gate.py \
    --circuit S38417 \
    --gate    XNR4D0BWP \
    --device  cuda \
    --gpu-extract \
    --baseline
```

```
============================================================
  S38417  ·  find all 'XNR4D0BWP'  ·  threshold 0.5  ·  cuda
============================================================
  Search space  : 983/30656 transistors kept  (3.2% of chip)
  GNN seeds     : 288 regions flagged  (K=4)
  Instances     : 2 found  ->  S38417_XNR4D0BWP.v
  --------------------------------------------------------
  GNN screen (cuda)                : 7.7842 s
      breakdown: reachability 0.42s  edges_encode 6.81s  seed_expand 0.33s
  VF3 confirm (carved sub-circuit) : 0.0041 s
  Total                            : 7.7883 s
  --------------------------------------------------------
  Recall  (vs full VF3) : 2/2  (100%)
  Precision             : 2/2  (100%)
  --------------------------------------------------------
  Baseline: plain VF3 for 'XNR4D0BWP' over whole chip : 9.0737 s
  Speedup vs baseline : 1.2x
============================================================
```

The result lands in `outputs/S38417_XNR4D0BWP.v`:

```verilog
// XNR4D0BWP instances found in S38417
// 2 instance(s)
	// MATCH XNR4D0BWP M28478 M28461 M28460 M28459 ... M28477
	// MATCH XNR4D0BWP M28568 M28551 M28550 M28549 ... M28567
```

To see which gates a circuit contains (rarest first — those prune best):

```bash
grep "// MATCH" data/vf3_out/S38417.v | awk '{print $3}' | sort | uniq -c | sort -n | head
```

### Extract the complete gate-level netlist

```bash
python src/inference.py --circuit C432        # -> outputs/C432.v
```

### Useful flags for `find_gate.py`

| Flag | Meaning |
|---|---|
| `--gate G` | Target cell type (required) |
| `--threshold τ` | Screener keep-threshold, default `0.5`. Lower ⇒ more recall, larger search space |
| `--device {auto,cuda,cpu}` | Where the model runs |
| `--gpu-extract` | Run region extraction as sparse-tensor ops on the GPU **(recommended — see below)** |
| `--workers N` | CPU processes for extraction (only used *without* `--gpu-extract`) |
| `--baseline` | Also time plain VF3 for this gate over the whole chip, and print the head-to-head |

---

## Repository layout

```
├── src/
│   ├── parser.py         SPICE (.sp) → heterogeneous graph JSON
│   ├── extractor.py      VF3-based ground truth + K-hop dataset construction
│   ├── model.py          CircuitFilterGNN — target-conditioned R-GCN
│   ├── data_loader.py    Dataset / splits
│   ├── train.py          Training loop (early stopping, LR plateau)
│   ├── inference.py      FULL EXTRACTION: VF3 finds all gates, GNN annotates
│   └── find_gate.py      TARGETED SEARCH: GNN screens, VF3 confirms  ← the contribution
├── vf3_cpp/              VF3 exact matcher (parallel C++ implementation) + cell libraries
├── data/
│   ├── raw/              28 ISCAS-85/89 SPICE netlists (C17 … S38417)
│   ├── parsed/           Cached graph JSON
│   └── vf3_out/          Ground-truth gate-level Verilog from full-library VF3
├── checkpoints/          Trained model (best_model.pt)
├── configs/config.yaml   All hyperparameters
└── outputs/              Generated Verilog
```

---

## How the pipeline works

### 1. Netlist → graph (`parser.py`)

A SPICE netlist becomes a **heterogeneous directed graph**:

- **Nodes** are transistors *and* nets. Node types: `VDD`, `GND`, `signal`, `PMOS`, `NMOS` (one-hot, 5 channels).
- **Edges** encode *which terminal* connects to which net — gate, source, drain — so the graph preserves the electrical role of each connection, not merely adjacency. Signal nets additionally carry reverse edges. **22 relation types** in total.

This typed encoding is what lets the model tell an `AOI` apart from an `OAI` built from the same transistor count.

### 2. Ground truth (`extractor.py`)

VF3 runs with the full cell library and produces the true gate-level netlist for each circuit (cached in `data/vf3_out/`). Each match tells us exactly which transistors constitute which gate — these become the **positive** training regions.

### 3. Training data

For each positive gate instance, the *K*-hop neighbourhood around one of its transistors is extracted (*K* is chosen per gate from its own radius). Four flavours of **negatives** are generated so the model learns hard distinctions rather than trivial ones:

| Negative type | What it teaches |
|---|---|
| `random` | Arbitrary regions aren't gates |
| `partial` | *Part* of a gate is not the gate |
| `mutation` | A gate with an edge perturbed is not the gate |
| `others` | A *different* gate type is not this gate |

The `partial` and `others` negatives are what make this hard: they force the model to distinguish a real `AOI22` from a fragment of an `AOI32`.

### 4. The model (`model.py`)

`CircuitFilterGNN` — a **target-conditioned Relational GCN**:

- 2 R-GCN layers, 128 hidden channels, 22 relations, 8 bases (basis decomposition keeps the parameter count sane across relations)
- **221,409 parameters**
- The *candidate region* and the *target gate* are encoded by the **same** encoder, then concatenated and scored by an MLP → `P(region contains target gate)`.

Conditioning on the target is the key design choice: one model handles *any* gate in the library, including gates it was never explicitly trained to output, because the gate's own topology is an input rather than a class label.

### 5. Inference — two modes

**`inference.py`** (full extraction): VF3 recovers every gate; the GNN attaches a confidence `p_hat` to each. The GNN is an *annotator* here — remove it and the gates are identical.

**`find_gate.py`** (targeted search): the contribution.

```
   circuit + one target gate
             │
   ┌─────────▼──────────┐
   │ 1. GNN SCREEN      │  score the K-hop region around every transistor
   │    (GPU)           │  keep those scoring ≥ τ  → "seeds"
   └─────────┬──────────┘
   ┌─────────▼──────────┐
   │ 2. EXPAND & CARVE  │  grow each seed to its whole cell (through signal
   │                    │  nets, not power rails); emit a reduced .sp
   └─────────┬──────────┘
   ┌─────────▼──────────┐
   │ 3. VF3 CONFIRM     │  exact match, single-cell library, on the carved
   │                    │  sub-circuit only → guaranteed-correct instances
   └─────────┬──────────┘
             ▼
      Verilog + metrics
```

The screener may be wrong; **the output cannot be**, because VF3 has the final word. The GNN buys speed, never correctness.

---

## The investigation: does learned pruning pay off?

This is the part worth reading. The hypothesis was simple — *screen first, match less, go faster*. Testing it honestly required dismantling it three times.

> **Setup for everything below:** `S38417` (30,656 transistors, 4,805 gates, ~55 cell types) and `C432` (532 transistors, 91 gates). Exact-matcher numbers were measured with the VF3 binary in this repository. GNN screen times were measured on an RTX 4050 Laptop GPU.

### The bar to beat

Full-library VF3 on the whole chip:

```bash
./vf3_cpp/build/prog -l vf3_cpp/examples/lib/libs38417.sp -s data/raw/S38417.sp -o /tmp/full.v
grep -E "Runtime|Instances|Coverage" /tmp/full.v
```

```
// Runtime:   0.624 s
// Instances: 4805
// Coverage:  30656/30656 (100.00%)
```

**0.62 seconds for the entire chip, 100% coverage.** VF3 is fast because it matches the whole library *together* with **tiered deletion**: it finds large gates first, deletes those transistors, so the remaining graph shrinks before the smaller gates are searched.

Hold on to that mechanism. It is the reason for everything that follows.

### Wall 1 — Whole-chip screening has nothing to prune

The obvious design is: screen for *any* gate, hand the flagged regions to VF3.

It fails immediately, and the coverage line above says why. **100% of transistors already belong to some gate.** Ask "where are the gates?" and the honest answer is "everywhere." Measured on `C432`, whole-library screening keeps **532/532 transistors — 0% pruned.**

> It is asking *"where are the LEGO bricks?"* in a model built entirely of bricks. A screener can only save work if there is something to throw away.

### The pivot to single-target search

If "where is *any* gate?" is unprunable, the question must be narrowed to **"where is *this specific* gate?"** — because most of a chip genuinely *isn't* any particular rare gate. That is real empty space, and it is the only framing in which a screener can help.

This is also, not coincidentally, the task the source paper actually addresses: *target circuit matching*.

But splitting a library-wide problem into per-gate problems carries two hidden costs, and they must be measured before claiming any victory.

### Wall 2 — Splitting the library double-counts transistors

**Measured with VF3 alone, no machine learning involved** — so the effect cannot be blamed on the model. Run the full library once, versus running each of the 27 cell types separately and unioning the results (`C432`):

| | Gates found | Precision | Transistor claims |
|---|---|---|---|
| Full library, together | **91** | 100% | 532 / 532 = **1.00×** (clean partition) |
| Gate-by-gate, combined | **140** | **65%** | 780 / 532 = **1.47×** (overlaps!) |

A phantom `AN4D0BWP` claims transistors `m295, m296` that truly belong to a `CKND0BWP` inverter. A phantom `AOI22D0BWP` claims eight transistors of a real `AOI32D0BWP`.

**Why:** a *piece* of a large gate can look exactly like a small gate. VF3 prevents this by deleting transistors as it matches them; split the library and that safeguard vanishes. The result is not a valid netlist — it is overlapping, contradictory claims.

*This also answers the natural question,* **"why not just run the GNN on every gate and compare against full VF3?"** *— because the gate-by-gate answer is wrong before speed even enters the discussion.*

### Wall 3 — Splitting the library is dramatically slower

Tiered deletion is also the *source of VF3's speed*. Take it away — run one cell type against the full chip — and matching collapses. Measured on `S38417`:

| Single-cell VF3 run | Time |
|---|---|
| one inverter (`CKND0BWP`) | **105.7 s** |
| one `AOI211D0BWP` | 22.7 s |
| one `XNR4D0BWP` | 20.3 s |
| one `NR2D0BWP` | 12.6 s |

Across ~55 cell types that extrapolates to **≈2,000 s (~35 minutes)** — versus **0.62 s** for the same library run together. **A ~3,000× penalty**, purely for splitting the work.

So the "obvious comparison" — screener on all gates vs. full VF3 — is doubly dead: the result is *incorrect* (Wall 2) **and** the baseline it would have to beat is thousands of times faster (Wall 3).

### What does work: single rare-gate search

Restrict the claim to what the method is actually for — **one rare gate in a large netlist** — and it delivers. A single gate cannot double-claim against itself, so correctness is preserved.

`S38417`, target `XNR4D0BWP` (2 instances in 30,656 transistors):

| Metric | Result |
|---|---|
| Search space kept | **983 / 30,656 = 3.2%** (97% of the chip skipped) |
| Recall | **2/2 = 100%** |
| Precision | **2/2 = 100%** |
| VF3 confirm on carved circuit | **0.004 s** (from ~20 s) |
| GNN screen (RTX 4050) | 7.8 s |
| **Total** | **7.8 s** vs single-cell VF3 **9.1 s** → **1.2× faster** |

The pruning mechanism is real: the screener narrows a 30k-transistor hunt to 3% of the chip, loses nothing, and the exact confirm becomes essentially free.

### The verdict, and why

**But 1.2× is a win over the wrong opponent, and intellectual honesty requires saying so.**

The 9.1 s baseline is *single-cell* VF3 — the crippled configuration from Wall 3. Nobody would actually run that. The configuration a real user runs is **full-library VF3: 0.62 s for every gate in the chip**, from which the `XNR4` instances can simply be filtered out.

| Approach | Time | What you get |
|---|---|---|
| **Full-library VF3, then filter** | **0.62 s** | Every gate, including all XNR4s |
| GNN screen + VF3 confirm | 7.8 s | Only the XNR4s |
| Single-cell VF3 (the flattering baseline) | 9.1 s | Only the XNR4s |

**Full VF3 is ~12× faster than our method and returns strictly more information.** our hierarchical matcher is reported faster still (~0.024 s).

**The root cause, stated plainly:** the screener has a **fixed cost** — it must look at the whole chip once, no matter what. On these netlists that fixed cost (≈8 s, or ≈2 s with further optimisation) is *larger than the entire matching problem it is trying to accelerate* (0.62 s).

> It is a toll booth that costs more than the drive.

**Learned pruning can only pay when the exact matcher is itself the bottleneck** — when matching takes seconds to minutes, so that skipping 97% of the work saves more than the screener costs. **On ISCAS-scale netlists, that regime does not exist.** Exact matching is simply too fast to be worth accelerating.

This is a negative result, and it is stated here plainly rather than buried, because it is the honest conclusion the evidence supports.

---

## Engineering: making the screener fast

Getting the screener from *unusably slow* to *faster than its baseline* was the bulk of the engineering, and the profiling story is instructive.

The screen must build and encode a *K*-hop region around **every one of 30,656 transistors**. The naive implementation calls PyTorch Geometric's `k_hop_subgraph` once per transistor:

| Implementation | Screen time (S38417) | Why |
|---|---|---|
| `k_hop_subgraph` per transistor, threaded | **275 s** (CPU) / 48–62 s | Each call rescans all 195k edges to return a ~21-node neighbourhood — 30k × full-graph scans |
| Neighbour-list BFS, tensors per region | 62 s | Graph-walk fixed, but 30k separate `torch.tensor()` constructions dominate |
| Pure-Python BFS, tensors batched per chunk | 80 s (`bfs_logic` 49 s) | Tensor cost gone; the *Python graph-walk itself* is now the wall — and the GPU sits idle |
| Multiprocess BFS across CPU cores | ~sub-linear gain | VF3 is parallel C++; racing it on the CPU is a losing game |
| **Sparse-tensor extraction on GPU** (`--gpu-extract`) | **7.8 s** | No Python graph-walk at all |

The final approach eliminates CPU graph traversal entirely:

1. **Reachability by sparse matrix powers.** Build the adjacency matrix `A` as a sparse tensor and a sparse `[N × C]` indicator of the *C* candidate centres. Multiply *K* times — `torch.sparse.mm(A, reach)` — and the non-zeros of the result *are* the K-hop neighbourhoods of every centre, computed simultaneously on the GPU. (0.42 s for all 30,656 regions.)
2. **Induced edges by vectorised gather + `searchsorted`.** For every node in every region, gather its out-edges via a CSR-style pointer, then keep only those whose target lies in the *same* region — a binary search over sorted `(region, node)` keys. No loops.
3. **Chunking around the power rails.** `VDD` and `GND` appear in nearly every region and have out-degree ≈25,000. Expanding their edges globally before filtering allocates **12.4 GB** and OOMs. Chunking the edge construction by region bounds the intermediate and fixes it.

**Every one of these rewrites was verified bit-exact** against the reference `k_hop_subgraph` implementation — max absolute difference `0.00e+00` on all embeddings at *K* = 2, 3, 4. Speed was never bought with silent approximation.

Residual inefficiency, honestly flagged: because the power rails touch nearly everything, `edges_encode` still generates and discards a large volume of candidate edges. Special-casing `VDD`/`GND` (intersecting them against each small region rather than expanding their 25k edges) should cut the screen to roughly **2 s**. It is not implemented here.

---

## Model and training

| | |
|---|---|
| Architecture | Target-conditioned R-GCN, 2 layers, 128 hidden, 22 relations, 8 bases |
| Parameters | 221,409 |
| Training set | 25 circuits, 94 gate types → 11,107 positives + 27,832 negatives (38,939 regions) |
| Optimiser | Adam, lr 1e-3, weight decay 1e-4, grad-clip 1.0 |
| Schedule | ReduceLROnPlateau on val-F1; early stopping (patience 60) |
| Best checkpoint | **epoch 285** |

**Validation metrics at the selected checkpoint:**

| Metric | Value |
|---|---|
| **Recall** | **0.955** |
| Precision | 0.677 |
| F1 | 0.792 |
| Accuracy | 0.855 |

**Recall is the metric that matters here**, and it is deliberately optimised for. The screener's job is to *never miss* a true gate region — a missed region is a gate lost forever, since VF3 never looks there. False positives merely cost a little extra confirm time and are eliminated by VF3 downstream. **0.955 recall with VF3 as backstop is exactly the right operating point**, and it is why end-to-end recall on the targeted-search task comes out at 100%.

The moderate precision (0.677) reflects genuinely hard negatives: distinguishing a real gate from *a fragment of a larger gate* or *a structurally similar different gate* across 94 cell types. It is not directly comparable to the source paper's figures, which use easier randomly-sampled negatives.

---

## Reproducing every number in this README

```bash
chmod +x vf3_cpp/build/prog

# The bar: full-library VF3, whole chip  (0.62 s, 4805 gates, 100% coverage)
./vf3_cpp/build/prog -l vf3_cpp/examples/lib/libs38417.sp -s data/raw/S38417.sp -o /tmp/full.v
grep -E "Runtime|Instances|Coverage" /tmp/full.v

# Wall 1: whole-library screening prunes nothing
python src/inference_prune.py --circuit C432 --threshold 0.7      # keeps 532/532

# Wall 2: gate-by-gate double-counts (VF3 alone, no ML)
#   full library -> 91 gates, 1.00x claims
#   gate-by-gate -> 140 gates, 65% precision, 1.47x claims

# Wall 3: single-cell VF3 is slow (build a one-cell library, run it on the full chip)

# What works: single rare-gate targeted search
python src/find_gate.py --circuit S38417 --gate XNR4D0BWP --device cuda --gpu-extract --baseline

# Full extraction (the practical path)
python src/inference.py --circuit S38417
```

**Provenance of the figures.** Exact-matcher timings (0.62 s full-library; 12.6–105.7 s single-cell; the 91-vs-140 double-counting proof) were measured directly with the VF3 binary in this repository. GNN screen timings (7.8 s, 3.2 % pruning, 100 % recall/precision) were measured on an RTX 4050 Laptop GPU. The hierarchical-matcher figure (0.024 s) is quoted from a collaborator's benchmark table and was not independently reproduced here.

---

## Limitations and honest caveats

- **The method does not beat full-library VF3 at this scale.** This is the headline finding, not a footnote. See [the verdict](#the-verdict-and-why).
- **Timings are hardware-dependent.** The screen was measured on a laptop GPU; the exact matcher on a laptop CPU under WSL. Absolute numbers will shift, but the *ratio* — screener fixed cost ≫ total matching cost — is structural, not an artefact.
- **The screener helps only for rare targets.** Common gates (e.g. inverters, 4,585 instances) prune poorly (75 % of the chip retained) and additionally suffer low single-cell precision (47 %), because a simple pattern occurs inside many larger gates.
- **Precision without VF3 would be unacceptable.** The GNN alone is a ~0.68-precision predictor. The exactness of the output depends entirely on VF3 confirming every candidate. This is by design, and it should not be mistaken for the GNN being accurate on its own.
- **The `.v` output is a match list**, not a fully elaborated structural Verilog module with pin-to-net bindings.
- **Untested above 30k transistors.** Every conclusion here is scoped to ISCAS-85/89. The regime where this method *could* win is explicitly the one not yet measured.

---

## Future work

The single decisive experiment this project points to:

**Does the crossover exist?** Learned pruning wins only when exact matching is slow. It is slow only on much larger designs. So:

1. Scale to **hundreds of thousands to millions of gates** — the large EPFL benchmarks (`multiplier`, `divisor`, `hypotenuse` ≈ 214k AND-nodes), ITC'99 (`b18`, `b19`), or an industrial netlist. These ship as gate-level Verilog/BLIF and must first be technology-mapped and expanded to transistor level to match the format here.
2. Benchmark GNN-screen + VF3 against **the strongest available matchers** — full-library VF3 *and* the hierarchical matcher — across many target gates.
3. **Characterise the crossover:** at what netlist size, and for what target rarity, does the screener's fixed cost finally become worth paying? That curve is the real scientific contribution — and it may show the crossover is unreachable, which would be an equally valuable answer.

Secondary items: implement the `VDD`/`GND` special-case (≈8 s → ≈2 s); emit fully structural Verilog with pin bindings; sweep the keep-threshold τ to trace the recall/pruning trade-off.

---

## References and license

**Method reproduced:**
Seo, Seo, Lee, Kim, Shin, Sung, Park. *Target Circuit Matching in Large-Scale Netlists Using GNN-Based Region Prediction.* arXiv:2507.19518, 2025.
(improvised through bfs verification) 
**Exact matcher:**
Carletti, Foggia, Saggese, Vento. *Introducing VF3: A New Algorithm for Subgraph Isomorphism.* GbRPR 2017. (Parallel C++ implementation vendored in `vf3_cpp/`.)

**Benchmarks:** ISCAS-85 and ISCAS-89 combinational/sequential benchmark circuits.

Released under the MIT License. © 2026 usuallyarnav.
