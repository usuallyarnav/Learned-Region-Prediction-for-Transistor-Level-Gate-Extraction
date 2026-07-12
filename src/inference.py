"""
GNN-as-search-space-pruner inference  (GPU + parallel).

Data flow (GNN is the workhorse, VF3 is the final verifier):

  1. GNN PRUNE  (timed) : for every gate type in the library, score the K-hop
                          region around every transistor in circuit C. Centers
                          scoring >= --threshold are kept; each is expanded to
                          its whole standard cell -> candidate set g ⊆ C.
                          Candidate-region encodings are computed ONCE per
                          distinct K and reused across all gates of that K.
  2. CARVE      (untimed): write a reduced .sp with only g's transistors.
  3. VF3 MATCH  (timed) : VF3 on the reduced circuit as the final exact matcher.
                          --target G[,G2,...] -> library restricted to those
                          cells; otherwise the full library.
  4. REPORT             : pruning ratio, the two timings, and recall/precision
                          vs the full-VF3 ground truth in data/vf3_out/<name>.v.

GPU / parallel:
  --device {auto,cuda,cpu}   model + all batches run on this device (auto=cuda
                             if available). The GNN math is the only part that
                             benefits from the GPU.
  --workers N                parallelise the K-hop region extraction (CPU, the
                             serial bottleneck on large netlists) across N
                             threads feeding the GPU encoder. 0 = serial.
  --no-power-expand          when expanding a kept seed to its whole cell, do
                             NOT travel through VDD/GND. Power rails touch every
                             transistor, so the default K-hop ball leaks across
                             the whole chip; cells connect through SIGNAL nets,
                             so this keeps whole gates while cutting the leak.
                             (Scoring still uses the full graph — only the
                             expansion is restricted.)

Drop-in sibling of inference.py; imports only existing project modules.
Author: usuallyarnav (MIT).
"""

import argparse
import contextlib
import io
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch
from torch_geometric.utils import k_hop_subgraph

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

import parser as spice_parser
from model import CircuitFilterGNN
from extractor import _build_full_graph, _extract_subgraph, target_radius, _run_vf3

_MATCH_RE = re.compile(r"^\s*//\s*MATCH\s+(\S+)\s+(.*)$")
_RUNTIME_RE = re.compile(r"//\s*Runtime:\s*([0-9.]+)")
_COVERAGE_RE = re.compile(r"//\s*Coverage:\s*(.+)")
_MLINE_RE = re.compile(r"^\s*([Mm]\S+)\s+\S")


# ── setup (untimed) ──────────────────────────────────────────────────────────
def pick_device(arg):
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        if not torch.cuda.is_available():
            sys.exit("[ERROR] --device cuda but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(config, checkpoint_path, device):
    m = config["model"]
    model = CircuitFilterGNN(in_channels=m["in_channels"], hidden_channels=m["hidden_channels"],
                             num_relations=m["num_relations"], num_bases=m["num_bases"],
                             num_layers=m["num_layers"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    return model.to(device).eval()


def parse_spice_dict(text):
    with contextlib.redirect_stdout(io.StringIO()):
        return spice_parser.parse_spice_to_heterogeneous_graph(text, "/tmp/_prune_tmp.json")


def build_targets(lib_text, fallback_k):
    blocks = spice_parser.split_library_into_subckts(lib_text)
    targets = {}
    for name, block in blocks.items():
        g = _build_full_graph(parse_spice_dict(block))
        if g.num_nodes == 0:
            continue
        targets[name] = (g, target_radius(g, fallback_k))
    return targets


def id_to_bare(json_dict):
    return {int(k): v.split("/")[-1].lower() for k, v in json_dict["id_to_node_name"].items()}


# ── phase 1: GNN region prediction → candidate set g (TIMED) ─────────────────
def _extract_many(full_graph, centers, k, workers):
    """K-hop region Data for each center; optionally extracted in parallel threads."""
    if workers and workers > 1 and len(centers) > workers:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda c: _extract_subgraph(full_graph, int(c), k), centers))
    return [_extract_subgraph(full_graph, int(c), k) for c in centers]


@torch.inference_mode()
def gnn_keep_centers(model, full_graph, targets, gate_types, centers, threshold,
                     batch_size, device, workers):
    """Returns {gate: set(center_id)} for centers scoring >= threshold.
    Candidate regions are encoded once per distinct K and reused across gates."""
    gates_by_k = defaultdict(list)
    for g in gate_types:
        if g in targets:
            gates_by_k[targets[g][1]].append(g)

    kept = {}
    for k, gates in gates_by_k.items():
        regions = _extract_many(full_graph, centers, k, workers)
        h_chunks = []
        for start in range(0, len(regions), batch_size):
            batch = Batch.from_data_list(regions[start:start + batch_size]).to(device)
            h_chunks.append(model.encode(batch))
        h_cand = torch.cat(h_chunks, dim=0)                          # [N, D] shared per K
        for g in gates:
            h_t = model.encode(Batch.from_data_list([targets[g][0]]).to(device))
            logit = model.mlp(torch.cat([h_cand, h_t.expand(h_cand.size(0), -1)], dim=1)).squeeze(1)
            idx = (torch.sigmoid(logit) >= threshold).nonzero(as_tuple=True)[0].tolist()
            kept[g] = {int(centers[i]) for i in idx}
        del h_cand
    return kept


def signal_edge_index(full_graph):
    """Edges with neither endpoint a power net (node types 0=VDD, 1=GND)."""
    nt = full_graph.node_types
    ei = full_graph.edge_index
    is_pwr = (nt == 0) | (nt == 1)
    keep = ~(is_pwr[ei[0]] | is_pwr[ei[1]])
    return ei[:, keep]


def expand_to_cells(full_graph, kept, targets, exp_edge_index):
    """Expand each kept center to the transistors in its K-hop region (whole cell),
    unioned across gates. exp_edge_index controls whether expansion crosses power."""
    nt = full_graph.node_types
    cand = set()
    for g, cs in kept.items():
        K = targets[g][1]
        for c in cs:
            subset, _, _, _ = k_hop_subgraph(
                int(c), K, exp_edge_index,
                relabel_nodes=False, num_nodes=full_graph.num_nodes,
            )
            for n in subset.tolist():
                if nt[n] >= 3:
                    cand.add(n)
    return cand


# ── carve (untimed) ──────────────────────────────────────────────────────────
def write_reduced_circuit(circuit_text, keep_bare, out_path):
    out, kept_n, total_n = [], 0, 0
    for line in circuit_text.splitlines():
        m = _MLINE_RE.match(line)
        if m:
            total_n += 1
            if m.group(1).lower() in keep_bare:
                out.append(line); kept_n += 1
        else:
            out.append(line)
    Path(out_path).write_text("\n".join(out) + "\n")
    return kept_n, total_n


def write_subset_lib(lib_text, gates, out_path):
    blocks = spice_parser.split_library_into_subckts(lib_text)
    missing = [g for g in gates if g not in blocks]
    if missing:
        raise SystemExit(f"[ERROR] gate(s) not in library: {', '.join(missing)} "
                         f"(available: {', '.join(sorted(blocks))})")
    Path(out_path).write_text("\n".join(blocks[g].strip() for g in gates) + "\n")


# ── VF3 final match (TIMED) ──────────────────────────────────────────────────
def vf3_match(vf3_bin, lib_sp, circuit_sp, out_v):
    ok, msg = _run_vf3(vf3_bin, lib_sp, circuit_sp, out_v)
    if not ok:
        raise RuntimeError(f"VF3 failed: {msg}")
    runtime, coverage = None, None
    for line in Path(out_v).read_text().splitlines():
        if (r := _RUNTIME_RE.search(line)):
            runtime = float(r.group(1))
        if (c := _COVERAGE_RE.search(line)):
            coverage = c.group(1).strip()
    return runtime, coverage


# ── accuracy vs full-VF3 ground truth (untimed) ──────────────────────────────
def signatures(out_v, restrict=None):
    sigs = set()
    for line in Path(out_v).read_text().splitlines():
        m = _MATCH_RE.match(line)
        if not m:
            continue
        gate = m.group(1)
        if restrict and gate not in restrict:
            continue
        fets = frozenset(x.lower() for x in m.group(2).split())
        if fets:
            sigs.add((gate, fets))
    return sigs


def main():
    ap = argparse.ArgumentParser(description="GNN search-space pruner + VF3 final match (GPU/parallel).")
    ap.add_argument("--circuit", required=True, help="circuit name (data/raw/<name>.sp) or a .sp path")
    ap.add_argument("--lib", default=None, help="cell library .sp (default vf3_cpp/examples/lib/lib<circuit>.sp)")
    ap.add_argument("--target", default=None, help="prune+match one or more gate types (comma-separated); "
                                                   "omit to process the whole library")
    ap.add_argument("--threshold", type=float, default=0.5, help="GNN keep threshold (default 0.5)")
    ap.add_argument("--no-power-expand", action="store_true",
                    help="expand cells through signal nets only, not VDD/GND (much tighter pruning)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int, default=0, help="threads for K-hop extraction (0=serial)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--config", default=str(_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=str(_ROOT / "outputs"))
    args = ap.parse_args()

    device = pick_device(args.device)
    config = yaml.safe_load(open(args.config))
    ke = config["extractor"]
    ckpt = args.checkpoint or (_ROOT / config["inference"]["checkpoint"])

    circuit_sp = Path(args.circuit)
    if not circuit_sp.exists():
        circuit_sp = _ROOT / "data" / "raw" / f"{args.circuit}.sp"
    circuit_name = circuit_sp.stem
    lib_sp = Path(args.lib) if args.lib else _ROOT / ke["lib_dir"] / f"lib{circuit_name.lower()}.sp"
    vf3_bin = _ROOT / ke["vf3_bin"]
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    out_v = out_dir / f"{circuit_name}_pruned.v"

    for p, what in [(circuit_sp, "circuit"), (lib_sp, "library"), (vf3_bin, "VF3 binary")]:
        if not Path(p).exists():
            sys.exit(f"[ERROR] {what} not found: {p}")

    model = load_model(config, ckpt, device)
    circuit_text = Path(circuit_sp).read_text(errors="replace")
    lib_text = Path(lib_sp).read_text(errors="replace")
    circuit_dict = parse_spice_dict(circuit_text)
    full_graph = _build_full_graph(circuit_dict)
    targets = build_targets(lib_text, ke["k_hops"])
    bare = id_to_bare(circuit_dict)

    requested = [t.strip() for t in args.target.split(",")] if args.target else sorted(targets)
    bad = [t for t in requested if t not in targets]
    if bad:
        sys.exit(f"[ERROR] target(s) not in library: {', '.join(bad)} "
                 f"({', '.join(sorted(targets))})")
    gate_types = requested

    nt = full_graph.node_types
    transistors = (nt >= 3).nonzero(as_tuple=True)[0].tolist()
    exp_ei = signal_edge_index(full_graph) if args.no_power_expand else full_graph.edge_index

    # ── PHASE 1 — GNN prune (TIMED) ──────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    kept = gnn_keep_centers(model, full_graph, targets, gate_types, transistors,
                            args.threshold, args.batch_size, device, args.workers)
    cand_nodes = expand_to_cells(full_graph, kept, targets, exp_ei)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_prune = time.perf_counter() - t0

    keep_bare = {bare[n] for n in cand_nodes if n in bare}
    n_pred_centers = sum(len(v) for v in kept.values())

    # ── CARVE (untimed) ──────────────────────────────────────────────────────
    reduced_sp = Path(tempfile.gettempdir()) / f"_{circuit_name}_pruned.sp"
    kept_n, total_n = write_reduced_circuit(circuit_text, keep_bare, reduced_sp)

    if args.target:
        used_lib = Path(tempfile.gettempdir()) / f"_lib_{'_'.join(gate_types)[:40]}.sp"
        write_subset_lib(lib_text, gate_types, used_lib)
    else:
        used_lib = lib_sp

    # ── PHASE 2 — VF3 final match on the pruned circuit (TIMED) ───────────────
    if kept_n == 0:
        t_match, coverage = 0.0, "0/0 (pruned to empty)"
        out_v.write_text("// pruned circuit empty — GNN kept no candidates\n")
    else:
        t1 = time.perf_counter()
        vf3_runtime, coverage = vf3_match(vf3_bin, used_lib, reduced_sp, out_v)
        t_match = vf3_runtime if vf3_runtime is not None else (time.perf_counter() - t1)

    # ── ACCURACY vs full-VF3 ground truth (untimed) ──────────────────────────
    truth_v = _ROOT / "data" / "vf3_out" / f"{circuit_name}.v"
    recall = precision = None
    n_truth = n_found = n_hit = 0
    restrict = set(gate_types) if args.target else None
    if truth_v.exists() and kept_n:
        truth = signatures(truth_v, restrict=restrict)
        found = signatures(out_v, restrict=restrict)
        hit = truth & found
        n_truth, n_found, n_hit = len(truth), len(found), len(hit)
        recall = n_hit / n_truth if n_truth else None
        precision = n_hit / n_found if n_found else None

    # ── REPORT ───────────────────────────────────────────────────────────────
    sep = "═" * 62
    scope = f"target {','.join(gate_types)}" if args.target else f"full library ({len(gate_types)} types)"
    prune_ratio = kept_n / total_n if total_n else 0.0
    print(f"\n{sep}")
    print(f"  {circuit_name} · {scope} · τ={args.threshold} · "
          f"{'signal-only' if args.no_power_expand else 'power-incl'} expand · {device.type}")
    print(sep)
    print(f"  Transistors kept (search space)    : {kept_n}/{total_n}  ({prune_ratio:.1%})")
    print(f"  GNN-predicted seed regions         : {n_pred_centers}")
    print(f"  ──────────────────────────────────────────────────────────")
    print(f"  Phase 1  GNN prune ({'cuda' if device.type=='cuda' else 'cpu'}, {args.workers or 1}w) : {t_prune:.6f} s")
    print(f"  Phase 2  VF3 final match (reduced)   : {t_match:.6f} s")
    print(f"  Total                                : {t_prune + t_match:.6f} s")
    print(sep)
    if recall is not None:
        print(f"  Recall  (vs full VF3)              : {n_hit}/{n_truth}   ({recall:.1%})")
        print(f"  Precision                          : {n_hit}/{n_found}   ({precision:.1%})")
    else:
        print(f"  (no ground truth at {truth_v.name} for recall)")
    print(f"  Coverage (VF3 on reduced)          : {coverage}")
    print(f"  Verilog written                    : {out_v}\n")


if __name__ == "__main__":
    main()