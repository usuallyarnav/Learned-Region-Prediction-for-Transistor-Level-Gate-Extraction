import json
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

_SRC_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _SRC_DIR.parent

PARSED_DIR = _PROJECT_DIR / "data" / "parsed"

NUM_RELATIONS = 22

def load_circuit_graph(json_path: str) -> Data:

    resolved = Path(json_path)
    if not resolved.suffix:
        resolved = PARSED_DIR / f"{resolved.name}.json"
    elif not resolved.is_absolute():

        resolved = _PROJECT_DIR / resolved

    with open(resolved, 'r') as f:
        raw_data = json.load(f)

    num_nodes      = raw_data['num_nodes']
    raw_node_types = raw_data['node_type']

    assert len(raw_node_types) == num_nodes, "Node type array length mismatch with num_nodes."

    type_mapping = {0: 0, 1: 1, 2: 2, 8: 3, 9: 4}

    def _map_type(t):
        if t not in type_mapping:
            print(f"[WARN] Unknown node type {t} in JSON — defaulting to NMOS (4)")
        return type_mapping.get(t, 4)

    mapped_types = [_map_type(t) for t in raw_node_types]
    type_tensor  = torch.tensor(mapped_types, dtype=torch.long)
    x            = F.one_hot(type_tensor, num_classes=5).to(torch.float)

    edge_index = torch.tensor(raw_data['edge_index'], dtype=torch.long)
    edge_type  = torch.tensor(raw_data['edge_type'],  dtype=torch.long)

    if edge_index.numel() > 0:
        assert edge_index.max().item() < num_nodes,            "Edge index points to out-of-bounds node."
        assert edge_type.max().item() < NUM_RELATIONS,            f"Edge type exceeds pipeline maximum of {NUM_RELATIONS - 1}"

    pyg_graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)

    pyg_graph.num_relations = NUM_RELATIONS

    pyg_graph.id_to_node_name = {
        int(k): v for k, v in raw_data.get('id_to_node_name', {}).items()
    }

    pyg_graph.node_dimensions = {
        int(k): v for k, v in raw_data.get('node_dimensions', {}).items()
    }

    if edge_index.numel() > 0:
        all_endpoints = edge_index.flatten()
        touched = torch.zeros(num_nodes, dtype=torch.bool)
        touched[all_endpoints] = True
        isolated = (~touched).sum().item()

        pyg_graph.isolated_count = int(isolated)

        if isolated > 0:
            print(
                f"[WARN] {isolated} isolated node(s) in '{resolved.name}'"
                f" — check netlist connectivity."
            )
    else:
        pyg_graph.isolated_count = num_nodes

    return pyg_graph

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/data_loader.py <stem>")
        print("       python src/data_loader.py <path/to/graph.json>")
        sys.exit(1)

    arg = sys.argv[1]

    if not arg.endswith('.json') and not os.path.isfile(arg):
        input_file = str(PARSED_DIR / f"{Path(arg).stem}.json")
    else:
        input_file = arg

    if not os.path.isfile(input_file):
        print(f"[ERROR] Parsed JSON not found: '{input_file}'")
        print(f"        Expected location  : {PARSED_DIR}/")
        print(f"        Run parser.py first: python src/parser.py {Path(arg).stem}")
        sys.exit(1)

    graph = load_circuit_graph(input_file)

    mapped = graph.x.argmax(dim=1)
    n_pmos = (mapped == 3).sum().item()
    n_nmos = (mapped == 4).sum().item()
    n_nets = graph.num_nodes - n_pmos - n_nmos

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  PyG Data Object  →  Loaded from '{Path(input_file).name}'")
    print(sep)
    print(f"  Nodes             : {graph.num_nodes:>5}   ({n_nets} nets, {n_pmos + n_nmos} MOSFETs)")
    print(f"  MOSFETs           :          PMOS={n_pmos}  NMOS={n_nmos}")
    print(f"  Node features (X) : {list(graph.x.shape)}")
    print(f"  Edge index        : {list(graph.edge_index.shape)}")
    print(f"  Pipeline Relations: {graph.num_relations:>5}   (Static Constant)")
    print(f"  Isolated nodes    : {graph.isolated_count:>5}")

    if graph.isolated_count > 0:
        print(f"  [WARN] {graph.isolated_count} completely detached node(s) detected.")
    print(f"{sep}\n")