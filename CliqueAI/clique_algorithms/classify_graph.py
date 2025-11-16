import os
import shutil

saved_graph_dir = "saved_graph"
dirs = {
    "0.1": os.path.join(saved_graph_dir, "0.1"),
    "0.2": os.path.join(saved_graph_dir, "0.2"),
    "0.4": os.path.join(saved_graph_dir, "0.4"),
}

# Ensure target directories exist
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

for fname in os.listdir(saved_graph_dir):
    if not fname.endswith(".clq"):
        continue
    fpath = os.path.join(saved_graph_dir, fname)

    # Read node number from file
    node_num = None
    with open(fpath, "r") as f:
        for line in f:
            if line.startswith("p edge"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    node_num = int(parts[2])
                break

    if node_num is None:
        print(f"Could not read node number in {fname}, skipping.")
        continue

    # Decide subdirectory
    if node_num <= 100:
        target_dir = dirs["0.1"]
    elif node_num <= 300:
        target_dir = dirs["0.2"]
    else:
        target_dir = dirs["0.4"]

    target_path = os.path.join(target_dir, fname)
    shutil.move(fpath, target_path)
    print(f"Moved {fname} (nodes: {node_num}) to {target_dir}")
