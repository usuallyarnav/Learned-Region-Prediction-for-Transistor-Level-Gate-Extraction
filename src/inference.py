
import argparse
import contextlib
import io
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

import parser as spice_parser
from model import CircuitFilterGNN
from extractor import _load_json, _build_full_graph, _extract_subgraph, target_radius, _run_vf3

_RUNTIME_RE = re.compile(r"//\s*Runtime:\s*([0-9.]+)")
_COVERAGE_RE = re.compile(r"//\s*Coverage:\s*(.+)")
_MATCH_LINE = re.compile(r"^\s*//\s*MATCH\s+(\S+)\s+(.*)$")


# ── setup helpers (untimed) ──────────────────────────────────────────────────
def load_model(config, checkpoint_path):
    m = config["model"]
    model = CircuitFilterGNN(in_channels=m["in_channels"], hidden_channels=m["hidden_channels"],
                             num_relations=m["num_relations"], num_bases=m["num_bases"],
                             num_layers=m["num_layers"])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    return model


def parse_spice_dict(text):
    with contextlib.redirect_stdout(io.StringIO()):
        return spice_parser.parse_spice_to_heterogeneous_graph(text, "/tmp/_infer_tmp.json")


def build_targets(lib_text, fallback_k):
    """Split the library into per-gate target graphs. Returns {gate: (graph, K)}."""
    blocks = spice_parser.split_library_into_subckts(lib_text)
    targets = {}
    for name, block in blocks.items():
        g = _build_full_graph(parse_spice_dict(block))
        if g.num_nodes == 0:
            continue
        targets[name] = (g, target_radius(g, fallback_k))
    return targets


def bare_name_to_id(json_dict):
    return {v.split("/")[-1].lower(): int(k) for k, v in json_dict["id_to_node_name"].items()}


# ── phase 1: VF3 exact matching (TIMED) ──────────────────────────────────────
def vf3_match(vf3_bin, lib_sp, circuit_sp, out_v):
    ok, msg = _run_vf3(vf3_bin, lib_sp, circuit_sp, out_v)
    if not ok:
        raise RuntimeError(f"VF3 failed: {msg}")
    runtime, coverage, instances, prev = None, None, [], ""
    for line in Path(out_v).read_text().splitlines():
        if (r := _RUNTIME_RE.search(line)):
            runtime = float(r.group(1))
        if (c := _COVERAGE_RE.search(line)):
            coverage = c.group(1).strip()
        if (m := _MATCH_LINE.match(line)):
            instances.append((m.group(1), m.group(2).split(), prev.strip()))
        elif line.strip():
            prev = line
    return runtime, coverage, instances


# ── phase 2: GNN region prediction (TIMED) ───────────────────────────────────
@torch.no_grad()
def score_matches(model, full_graph, targets, instances, bare_to_id, batch_size=256):
    """
    Default fast path: the output only reads GNN scores at matched transistors, so
    encode ONLY those regions. Region encoding depends only on K (not the target),
    so group the needed centers by K and encode in big batches, then score per
    gate. Output is identical to a full scan. Returns {gate: {center_id: prob}}.
    """
    needed = defaultdict(set)
    for gate, fets, _ in instances:
        if gate not in targets:
            continue
        for f in fets:
            if f.lower() in bare_to_id:
                needed[gate].add(bare_to_id[f.lower()])

    centers_by_k = defaultdict(set)
    for gate, centers in needed.items():
        centers_by_k[targets[gate][1]] |= centers

    hcand = {}                                                    # (k, center) -> embedding
    for k, centers in centers_by_k.items():
        centers = sorted(centers)
        embs = []
        for start in range(0, len(centers), batch_size):
            chunk = centers[start:start + batch_size]
            regions = [_extract_subgraph(full_graph, int(c), k) for c in chunk]
            embs.append(model.encode(Batch.from_data_list(regions)))
        H = torch.cat(embs, dim=0)
        for i, c in enumerate(centers):
            hcand[(k, c)] = H[i]

    scores = {}
    for gate, centers in needed.items():
        k = targets[gate][1]
        cs = sorted(centers)
        H = torch.stack([hcand[(k, c)] for c in cs], dim=0)       # [m, D]
        h_t = model.encode(Batch.from_data_list([targets[gate][0]]))
        p = torch.sigmoid(model.mlp(torch.cat([H, h_t.expand(H.size(0), -1)], dim=1))).squeeze(1)
        scores[gate] = {c: float(p[i]) for i, c in enumerate(cs)}
    return scores


@torch.no_grad()
def predict_regions(model, full_graph, targets, gate_types, centers, batch_size=256):
    """
    Full region prediction (--scan-all): score the K-hop region of every center
    for each gate type. Candidate-region encodings are computed once per distinct
    K and shared across all gates with that K. Returns {gate: {center_id: prob}}.
    """
    gates_by_k = defaultdict(list)
    for g in gate_types:
        if g in targets:
            gates_by_k[targets[g][1]].append(g)

    scores = {}
    for k, gates in gates_by_k.items():
        h_chunks = []
        for start in range(0, len(centers), batch_size):
            regions = [_extract_subgraph(full_graph, int(c), k) for c in centers[start:start + batch_size]]
            h_chunks.append(model.encode(Batch.from_data_list(regions)))
        h_cand = torch.cat(h_chunks, dim=0)                       # [N, D], shared per K
        for g in gates:
            h_t = model.encode(Batch.from_data_list([targets[g][0]]))
            p = torch.sigmoid(model.mlp(torch.cat([h_cand, h_t.expand(h_cand.size(0), -1)], dim=1))).squeeze(1)
            scores[g] = {int(centers[i]): float(p[i]) for i in range(len(centers))}
    return scores


# ── output (untimed) ─────────────────────────────────────────────────────────
def module_header(out_v, circuit):
    for line in Path(out_v).read_text().splitlines():
        if line.startswith("module "):
            return f"module {circuit}({line[line.index('(') + 1:]}"
    return f"module {circuit}();"


def write_verilog(path, header, annotated, coverage, t_infer, t_match):
    gate_types = sorted({g for _, g, _, _ in annotated})
    lines = [header]
    for phat, gate, fets, inst_line in annotated:
        score = "n/a" if phat is None else f"{phat:.4f}"
        lines.append(f"\t{inst_line}\t// p_hat={score}")
        lines.append(f"\t// MATCH {gate} {' '.join(fets)}")
    lines.append("endmodule")
    lines += ["",
              f"// Instances:        {len(annotated)}",
              f"// Gate types:       {len(gate_types)}",
              f"// Coverage:         {coverage if coverage else 'n/a'}",
              f"// Inference (GNN):  {t_infer:.6f} s",
              f"// Exact match (VF3):{t_match:.6f} s"]
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="GNN region prediction + VF3 whole-library gate extraction.")
    ap.add_argument("--circuit", required=True, help="circuit name (data/raw/<name>.sp) or a .sp path")
    ap.add_argument("--lib", default=None, help="cell library .sp (default: vf3_cpp/examples/lib/lib<circuit>.sp)")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=str(_ROOT / "outputs"))
    ap.add_argument("--centers", choices=["all", "transistors"], default="all",
                    help="centers for --scan-all region prediction (paper: all)")
    ap.add_argument("--scan-all", action="store_true",
                    help="full region prediction over every node (slower); default scores only matched regions")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    # ── SETUP (untimed) ──────────────────────────────────────────────────────
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
    out_v = out_dir / f"{circuit_name}.v"
    vf3_tmp = Path(tempfile.gettempdir()) / f"_{circuit_name}_vf3_full.v"

    for p, what in [(circuit_sp, "circuit"), (lib_sp, "library"), (vf3_bin, "VF3 binary")]:
        if not Path(p).exists():
            sys.exit(f"[ERROR] {what} not found: {p}")

    model = load_model(config, ckpt)
    circuit_dict = parse_spice_dict(Path(circuit_sp).read_text(errors="replace"))
    full_graph = _build_full_graph(circuit_dict)
    targets = build_targets(Path(lib_sp).read_text(errors="replace"), ke["k_hops"])
    bare_to_id = bare_name_to_id(circuit_dict)

    # ── PHASE 1 — VF3 exact matching (TIMED) ─────────────────────────────────
    t1 = time.perf_counter()
    vf3_runtime, coverage, instances = vf3_match(vf3_bin, lib_sp, circuit_sp, vf3_tmp)
    t_match_wall = time.perf_counter() - t1
    t_match = vf3_runtime if vf3_runtime is not None else t_match_wall

    found_gates = {g for g, _, _ in instances}

    # ── PHASE 2 — GNN region prediction (TIMED) ──────────────────────────────
    t0 = time.perf_counter()
    if args.scan_all:
        if args.centers == "transistors":
            centers = (full_graph.node_types >= 3).nonzero(as_tuple=True)[0].tolist()
        else:
            centers = list(range(full_graph.num_nodes))
        scores = predict_regions(model, full_graph, targets, found_gates, centers, args.batch_size)
        n_regions = f"{len(found_gates)} gate types x {len(centers)} regions"
    else:
        scores = score_matches(model, full_graph, targets, instances, bare_to_id, args.batch_size)
        n_regions = f"{sum(len(g) for g in scores.values())} matched regions"
    t_infer = time.perf_counter() - t0

    # ── ASSEMBLE OUTPUT (untimed) — preserve VF3 order, annotate p_hat ────────
    annotated = []
    for gate, fets, inst_line in instances:
        ids = [bare_to_id[f.lower()] for f in fets if f.lower() in bare_to_id]
        gmap = scores.get(gate, {})
        phat = max((gmap.get(i, 0.0) for i in ids), default=None) if gmap else None
        annotated.append((phat, gate, fets, inst_line))

    header = module_header(vf3_tmp, circuit_name)
    write_verilog(out_v, header, annotated, coverage, t_infer, t_match)

    # ── REPORT — only the two times we care about ────────────────────────────
    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  {circuit_name}  ←  library {lib_sp.name}")
    print(sep)
    print(f"  Inference  (GNN region scoring)    : {t_infer:.6f} s   [{n_regions}]")
    print(f"  Exact match (VF3)                  : {t_match:.6f} s")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  Total (inference + matching)       : {t_infer + t_match:.6f} s")
    print(sep)
    print(f"  Instances extracted                : {len(annotated)}  ({len(found_gates)} gate types)")
    print(f"  Coverage (VF3)                     : {coverage if coverage else 'n/a'}")
    print(f"  Verilog written                    : {out_v}")
    print(f"  (parsing, library split, model load, I/O excluded from timing)\n")


if __name__ == "__main__":
    main()