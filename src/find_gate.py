"""
find_gate.py  —  find every instance of ONE logic gate in a circuit.

This is the distilled, validated approach: the GNN narrows the search, VF3
confirms.  It does one thing only — targeted search for a single gate type G —
which is the regime where the learned model actually beats the exact checker.

    1.  GNN screen : score the K-hop region around every transistor against G,
                     keep the ones that look likely  (the "candidate seeds").
    2.  Expand     : grow each seed to its whole standard cell (through signal
                     nets, not power rails) and carve out that small sub-circuit.
    3.  VF3 confirm: run the exact checker with a one-cell library {G} on the
                     carved sub-circuit — this is the final, trustworthy match.
    4.  Report     : write a clean Verilog of G's instances, and print how much
                     of the chip was skipped, recall/precision vs the full-VF3
                     ground truth, and (with --baseline) the head-to-head speed
                     vs running VF3 for G over the whole chip.

Run:
    python src/find_gate.py --circuit S38417 --gate XNR4D0BWP --device cuda --workers 8 --baseline

For full-netlist extraction (find *every* gate) do NOT use this — the exact
checker alone is faster and cleaner there.  This script is for sparse targets.

Author: usuallyarnav (MIT).  Uses only existing project modules.
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph   # used only for seed->cell expansion

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

import parser as spice_parser
from model import CircuitFilterGNN
from extractor import _build_full_graph, target_radius, _run_vf3

_MATCH = re.compile(r"^\s*//\s*MATCH\s+(\S+)\s+(.*)$")
_RUNTIME = re.compile(r"//\s*Runtime:\s*([0-9.]+)")
_MLINE = re.compile(r"^\s*([Mm]\S+)\s+\S")


def load_model(cfg, ckpt, device):
    m = cfg["model"]
    model = CircuitFilterGNN(m["in_channels"], m["hidden_channels"],
                             m["num_relations"], m["num_bases"], m["num_layers"])
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state", state))
    return model.to(device).eval()


def parse_graph(text):
    with contextlib.redirect_stdout(io.StringIO()):
        d = spice_parser.parse_spice_to_heterogeneous_graph(text, "/tmp/_fg.json")
    return d, _build_full_graph(d)


def build_target(lib_text, gate, fallback_k):
    blocks = spice_parser.split_library_into_subckts(lib_text)
    if gate not in blocks:
        sys.exit(f"[ERROR] gate '{gate}' not in library. Available:\n  "
                 + ", ".join(sorted(blocks)))
    _, g = parse_graph(blocks[gate])
    return g, target_radius(g, fallback_k), blocks[gate]


def build_adjacency(graph):
    """Out- and in-neighbour lists (with edge types), built once in O(E).
    k_hop_subgraph re-scans all E edges on every call; these lists let us walk
    only each region's small neighbourhood instead."""
    ei, et, N = graph.edge_index, graph.edge_type, graph.num_nodes
    out = [[] for _ in range(N)]
    inn = [[] for _ in range(N)]
    for a, b, t in zip(ei[0].tolist(), ei[1].tolist(), et.tolist()):
        out[a].append((b, t))
        inn[b].append((a, t))
    return out, inn


_ADJ = {}   # set in the parent before forking; workers inherit it copy-on-write


def _bfs_task(args):
    """Worker: BFS the K-hop predecessor region for each center in this task and
    return flat, chunk-batched arrays. Pure Python — no torch, no CUDA — so it is
    safe to fork and parallelises cleanly across CPU cores."""
    centers, K = args
    adj_in, adj_out = _ADJ["in"], _ADJ["out"]
    node_ids, src, dst, ety, bvec, off = [], [], [], [], [], 0
    for j, c in enumerate(centers):
        c = int(c); seen = {c}; frontier = [c]
        for _ in range(K):
            nxt = []
            for u in frontier:
                for v, _t in adj_in[u]:                # predecessors (matches k_hop_subgraph)
                    if v not in seen:
                        seen.add(v); nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        S = sorted(seen)
        loc = {n: i + off for i, n in enumerate(S)}
        for u in S:
            lu = loc[u]
            for v, t in adj_out[u]:                    # induced edges, original direction
                if v in seen:
                    src.append(lu); dst.append(loc[v]); ety.append(t)
        node_ids.extend(S); bvec.extend([j] * len(S)); off += len(S)
    return node_ids, src, dst, ety, bvec


def extract_regions(graph, centers, K, workers, task=1500):
    """Build all K-hop regions. Extraction is pure-Python graph walking (the
    bottleneck on slow CPUs), so it is parallelised across `workers` cores.
    Runs BEFORE the model is on the GPU, so fork and CUDA never collide.
    Returns (list-of-chunks, timings); each chunk = (node_ids, src, dst, ety, bvec)."""
    T = {}
    t = time.perf_counter()
    adj_out, adj_in = build_adjacency(graph)
    _ADJ["out"], _ADJ["in"] = adj_out, adj_in
    T["adjacency"] = time.perf_counter() - t

    tasks = [(list(centers[i:i + task]), K) for i in range(0, len(centers), task)]
    t = time.perf_counter()
    if workers and workers > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            chunks = pool.map(_bfs_task, tasks)
    else:
        chunks = [_bfs_task(a) for a in tasks]
    T["bfs_logic"] = time.perf_counter() - t
    return chunks, T


@torch.inference_mode()
def gpu_score_regions(model, graph, target_graph, K, centers, device, rchunk=400):
    """All-tensor extraction: sparse matrix powers for the K-hop neighbourhoods,
    vectorised gather + searchsorted for induced edges. No Python graph-walk, so
    the work runs on the GPU instead of the CPU. Bit-identical to the per-region
    path (verified 0.0 diff). Edge construction is chunked by regions because the
    power rails (VDD/GND, out-degree ~25k, in almost every region) would otherwise
    generate billions of candidate edges before filtering. Returns (prob, timings)."""
    T = {"reachability": 0.0, "edges_encode": 0.0}
    N = graph.num_nodes
    ei = graph.edge_index.to(device); et = graph.edge_type.to(device); C = len(centers)

    t = time.perf_counter()
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.size(1), device=device), (N, N)).coalesce()
    ct = torch.tensor(centers, device=device)
    reach = torch.sparse_coo_tensor(torch.stack([ct, torch.arange(C, device=device)]),
                                    torch.ones(C, device=device), (N, C)).coalesce()
    for _ in range(K):
        nxt = torch.sparse.mm(A, reach)
        reach = (reach + nxt).coalesce()
        reach = torch.sparse_coo_tensor(reach.indices(), torch.ones(reach.values().numel(), device=device), (N, C)).coalesce()
    ri = reach.indices(); order = torch.argsort(ri[1], stable=True)
    bn_node = ri[0][order]; bn_region = ri[1][order].to(torch.long)
    rptr = torch.zeros(C + 1, dtype=torch.long, device=device)
    rptr[1:] = torch.bincount(bn_region, minlength=C).cumsum(0)
    src = ei[0]; eo = torch.argsort(src); sdst = ei[1][eo]; setype = et[eo]
    out_ptr = torch.zeros(N + 1, dtype=torch.long, device=device)
    out_ptr[1:] = torch.bincount(src, minlength=N).cumsum(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    T["reachability"] = time.perf_counter() - t

    t = time.perf_counter()
    x = graph.x.to(device); h_chunks = []
    for a in range(0, C, rchunk):
        b = min(a + rchunk, C); plo = int(rptr[a]); phi = int(rptr[b]); nloc = phi - plo
        bn = bn_node[plo:phi]; br = bn_region[plo:phi] - a
        keys = br * N + bn.to(torch.long); perm = torch.argsort(keys); skeys = keys[perm]
        deg = out_ptr[bn + 1] - out_ptr[bn]; tot = int(deg.sum())
        cp = torch.repeat_interleave(torch.arange(nloc, device=device), deg)
        gs = torch.cumsum(deg, 0) - deg
        sidx = out_ptr[bn][cp] + (torch.arange(tot, device=device) - gs[cp])
        ctg = sdst[sidx]; cty = setype[sidx]
        qk = br[cp] * N + ctg.to(torch.long)
        pos = torch.searchsorted(skeys, qk).clamp(max=nloc - 1)
        val = skeys[pos] == qk
        d = Data(x=x[bn], edge_index=torch.stack([cp[val], perm[pos[val]]]), edge_type=cty[val])
        d.batch = br
        h_chunks.append(model.encode(d))
    h = torch.cat(h_chunks, dim=0)
    h_t = model.encode(Batch.from_data_list([target_graph]).to(device))
    logit = model.mlp(torch.cat([h, h_t.expand(h.size(0), -1)], dim=1)).squeeze(1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    T["edges_encode"] = time.perf_counter() - t
    return torch.sigmoid(logit), T


@torch.inference_mode()
def encode_chunks(model, graph, target_graph, chunks, device):
    """Build tensors once per chunk and encode on the GPU. Small chunks keep the
    RGCN forward fast (one huge graph is much slower). Returns (prob, timings)."""
    T = {"tensor_build": 0.0, "gpu_forward": 0.0}
    x_cpu = graph.x
    h_chunks = []
    for node_ids, src, dst, ety, bvec in chunks:
        t = time.perf_counter()
        nidx = torch.tensor(node_ids, dtype=torch.long)
        d = Data(x=x_cpu[nidx],
                 edge_index=torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long),
                 edge_type=torch.tensor(ety, dtype=torch.long) if ety else torch.zeros((0,), dtype=torch.long))
        d.batch = torch.tensor(bvec, dtype=torch.long)
        d = d.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        T["tensor_build"] += time.perf_counter() - t

        t = time.perf_counter()
        h_chunks.append(model.encode(d))
        if device.type == "cuda":
            torch.cuda.synchronize()
        T["gpu_forward"] += time.perf_counter() - t

    h = torch.cat(h_chunks, dim=0)
    h_t = model.encode(Batch.from_data_list([target_graph]).to(device))
    logit = model.mlp(torch.cat([h, h_t.expand(h.size(0), -1)], dim=1)).squeeze(1)
    return torch.sigmoid(logit), T


def signal_edges(graph):
    """Edges with no power-net endpoint (node types 0=VDD, 1=GND)."""
    nt, ei = graph.node_types, graph.edge_index
    pwr = (nt == 0) | (nt == 1)
    return ei[:, ~(pwr[ei[0]] | pwr[ei[1]])]


def expand_to_cells(graph, seeds, K, exp_edges):
    """Grow each seed to the transistors in its K-hop region (whole cell)."""
    nt = graph.node_types
    keep = set()
    for c in seeds:
        subset, _, _, _ = k_hop_subgraph(int(c), K, exp_edges,
                                         relabel_nodes=False, num_nodes=graph.num_nodes)
        keep.update(n for n in subset.tolist() if nt[n] >= 3)
    return keep


def carve(circuit_text, keep_names, out_path):
    """Reduced .sp: every non-transistor line verbatim, only kept M-lines."""
    lines, kept, total = [], 0, 0
    for ln in circuit_text.splitlines():
        m = _MLINE.match(ln)
        if m:
            total += 1
            if m.group(1).lower() in keep_names:
                lines.append(ln); kept += 1
        else:
            lines.append(ln)
    Path(out_path).write_text("\n".join(lines) + "\n")
    return kept, total


def vf3(vf3_bin, lib_sp, circuit_sp, out_v):
    ok, msg = _run_vf3(vf3_bin, lib_sp, circuit_sp, out_v)
    if not ok:
        raise RuntimeError(f"VF3 failed: {msg}")
    rt = next((float(m.group(1)) for ln in Path(out_v).read_text().splitlines()
               if (m := _RUNTIME.search(ln))), None)
    return rt


def gate_signatures(out_v, gate):
    """Instances of `gate` as {frozenset(transistor names)}."""
    sigs = set()
    for ln in Path(out_v).read_text().splitlines():
        m = _MATCH.match(ln)
        if m and m.group(1) == gate:
            fets = frozenset(x.lower() for x in m.group(2).split())
            if fets:
                sigs.add(fets)
    return sigs


def write_clean_verilog(circuit_name, gate, out_v, dest):
    """Emit only `gate`'s instance + MATCH lines, with a small summary footer."""
    keep = [ln for ln in Path(out_v).read_text().splitlines()
            if gate in ln or _MATCH.match(ln) and _MATCH.match(ln).group(1) == gate]
    body = [ln for ln in Path(out_v).read_text().splitlines()
            if (m := _MATCH.match(ln)) and m.group(1) == gate]
    header = f"// {gate} instances found in {circuit_name}\n// {len(body)} instance(s)\n"
    Path(dest).write_text(header + "\n".join(body) + "\n")
    return len(body)


def main():
    ap = argparse.ArgumentParser(description="Find every instance of one gate (GNN prune -> VF3 confirm).")
    ap.add_argument("--circuit", required=True, help="name (data/raw/<name>.sp) or a .sp path")
    ap.add_argument("--gate", required=True, help="target gate type, e.g. XNR4D0BWP")
    ap.add_argument("--threshold", type=float, default=0.5, help="GNN keep threshold (default 0.5)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--workers", type=int, default=8, help="CPU worker processes for region extraction (default 4; set to your core count)")
    ap.add_argument("--gpu-extract", action="store_true", help="run region extraction as sparse-tensor ops on the GPU instead of CPU BFS (experimental)")
    ap.add_argument("--baseline", action="store_true", help="also time plain VF3 for G over the whole chip")
    ap.add_argument("--config", default=str(_ROOT / "configs" / "config.yaml"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=str(_ROOT / "outputs"))
    args = ap.parse_args()

    device = (torch.device("cuda") if (args.device == "cuda" or
              (args.device == "auto" and torch.cuda.is_available())) else torch.device("cpu"))
    if args.device == "cuda" and device.type != "cuda":
        sys.exit("[ERROR] --device cuda but no CUDA device available")

    cfg = yaml.safe_load(open(args.config))
    ke = cfg["extractor"]
    ckpt = args.checkpoint or (_ROOT / cfg["inference"]["checkpoint"])

    circuit_sp = Path(args.circuit)
    if not circuit_sp.exists():
        circuit_sp = _ROOT / "data" / "raw" / f"{args.circuit}.sp"
    name = circuit_sp.stem
    lib_sp = _ROOT / ke["lib_dir"] / f"lib{name.lower()}.sp"
    vf3_bin = _ROOT / ke["vf3_bin"]
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    for pth, what in [(circuit_sp, "circuit"), (lib_sp, "library"), (vf3_bin, "VF3 binary")]:
        if not Path(pth).exists():
            sys.exit(f"[ERROR] {what} not found: {pth}")

    # setup (untimed) — model loads on CPU so the fork pool never sees a CUDA context
    model = load_model(cfg, ckpt, torch.device("cpu"))
    circuit_text = Path(circuit_sp).read_text(errors="replace")
    lib_text = Path(lib_sp).read_text(errors="replace")
    circuit_dict, graph = parse_graph(circuit_text)
    target_graph, K, gate_block = build_target(lib_text, args.gate, ke["k_hops"])
    bare = {int(k): v.split("/")[-1].lower() for k, v in circuit_dict["id_to_node_name"].items()}
    transistors = (graph.node_types >= 3).nonzero(as_tuple=True)[0].tolist()

    # 1-2. GNN screen (timed)
    t0 = time.perf_counter()
    if args.gpu_extract:
        model = model.to(device)                                          # all-tensor path lives on device
        prob, screen_timings = gpu_score_regions(model, graph, target_graph, K, transistors, device)
    else:
        chunks, T_ex = extract_regions(graph, transistors, K, args.workers)   # parallel CPU, no CUDA
        model = model.to(device)                                              # NOW touch the GPU
        prob, T_en = encode_chunks(model, graph, target_graph, chunks, device)
        screen_timings = {**T_ex, **T_en}
    seeds = [transistors[i] for i in (prob >= args.threshold).nonzero(as_tuple=True)[0].tolist()]
    t_exp = time.perf_counter()
    cand = expand_to_cells(graph, seeds, K, signal_edges(graph))
    screen_timings["seed_expand"] = time.perf_counter() - t_exp
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_gnn = time.perf_counter() - t0

    keep_names = {bare[n] for n in cand if n in bare}
    reduced_sp = Path(tempfile.gettempdir()) / f"_find_{name}_{args.gate}.sp"
    kept, total = carve(circuit_text, keep_names, reduced_sp)

    # 3. VF3 confirm on the carved sub-circuit  (timed)
    one_cell = Path(tempfile.gettempdir()) / f"_lib_{args.gate}.sp"
    one_cell.write_text(gate_block.strip() + "\n")
    out_v = out_dir / f"{name}_{args.gate}_raw.v"
    if kept:
        t_vf3 = vf3(vf3_bin, one_cell, reduced_sp, out_v) or 0.0
    else:
        t_vf3 = 0.0; out_v.write_text("// GNN kept no candidates\n")

    # 4. clean output + accuracy vs full-VF3 ground truth
    clean = out_dir / f"{name}_{args.gate}.v"
    n_found = write_clean_verilog(name, args.gate, out_v, clean) if kept else 0
    truth_v = _ROOT / "data" / "vf3_out" / f"{name}.v"
    recall = precision = None; n_truth = n_hit = 0
    if truth_v.exists() and kept:
        truth = gate_signatures(truth_v, args.gate)
        found = gate_signatures(out_v, args.gate)
        n_truth, n_hit = len(truth), len(truth & found)
        recall = n_hit / n_truth if n_truth else None
        precision = n_hit / len(found) if found else None

    # optional head-to-head baseline: plain VF3 for G over the WHOLE chip
    base_t = None
    if args.baseline:
        base_out = Path(tempfile.gettempdir()) / f"_base_{name}_{args.gate}.v"
        base_t = vf3(vf3_bin, one_cell, circuit_sp, base_out) or 0.0

    # report
    sep = "=" * 60
    print(f"\n{sep}\n  {name}  ·  find all '{args.gate}'  ·  threshold {args.threshold}  ·  {device.type}\n{sep}")
    print(f"  Search space  : {kept}/{total} transistors kept  ({kept/total:.1%} of chip)")
    print(f"  GNN seeds     : {len(seeds)} regions flagged  (K={K})")
    print(f"  Instances     : {n_found} found  ->  {clean.name}")
    print(f"  {'-'*56}")
    print(f"  GNN screen ({device.type})             : {t_gnn:.4f} s")
    print(f"      breakdown: " + "  ".join(f"{k} {v:.2f}s" for k, v in screen_timings.items()))
    print(f"  VF3 confirm (carved sub-circuit) : {t_vf3:.4f} s")
    print(f"  Total                            : {t_gnn + t_vf3:.4f} s")
    if recall is not None:
        print(f"  {'-'*56}")
        print(f"  Recall  (vs full VF3) : {n_hit}/{n_truth}  ({recall:.0%})")
        print(f"  Precision             : {n_hit}/{len(found)}  ({precision:.0%})")
    if base_t is not None:
        speed = base_t / (t_gnn + t_vf3) if (t_gnn + t_vf3) else float("inf")
        print(f"  {'-'*56}")
        print(f"  Baseline: plain VF3 for '{args.gate}' over whole chip : {base_t:.4f} s")
        print(f"  Speedup vs baseline : {speed:.1f}x")
    print(sep + "\n")


if __name__ == "__main__":
    main()