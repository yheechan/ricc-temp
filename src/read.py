from pathlib import Path

script_file = Path(__file__).resolve()
curr_dir = script_file.parent
main_dir = curr_dir.parent

train_txt = main_dir / "train_0.txt"
with open(train_txt, "r") as f:
    lines = f.readlines()
    neg_nodes = lines[0].strip().split(" ")
    pos_nodes = lines[1].strip().split(" ")
    print(f"cnt of neg_nodes: {len(neg_nodes)}")
    print(f"cnt of pos_nodes: {len(pos_nodes)}")

target_close_txt = main_dir / "target_close.txt"
with open(target_close_txt, "r") as f:
    lines = f.readlines()
    target_close = lines[0].strip().split(" ")
    print(f"cnt of target_close: {len(target_close)}")

train_nodes = neg_nodes + pos_nodes
intrain, notintrain = 0, 0
for target_node in target_close:
    if target_node in train_nodes:
        intrain += 1
        print(f"target_node: {target_node} is in train_nodes")
    else:
        notintrain += 1
        print(f"target_node: {target_node} is not in train_nodes")

print(f"cnt of intrain: {intrain}")
print(f"cnt of notintrain: {notintrain}")