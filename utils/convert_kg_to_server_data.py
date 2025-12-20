import json
import re
from pathlib import Path


def _extract_version(name: str, prefix: str) -> int:
    """从目录名中提取版本号，匹配 prefix 后的整数，不存在则返回 -1。"""
    m = re.match(rf"{re.escape(prefix)}(\d+)$", name)
    return int(m.group(1)) if m else -1


def find_latest_kg_path() -> Path:
    """在 data 目录下找到 project_v* 中版本号最大的项目，再取其中迭代号最大的 knowledge_graph.json。"""
    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("data 目录不存在")

    project_dirs = [
        p for p in data_dir.iterdir()
        if p.is_dir() and _extract_version(p.name, "project_v") >= 0
    ]
    if not project_dirs:
        raise FileNotFoundError("未找到 project_v* 目录")

    project_dirs.sort(key=lambda p: _extract_version(p.name, "project_v"), reverse=True)
    latest_project = project_dirs[0]

    iteration_dirs = [
        p for p in latest_project.iterdir()
        if p.is_dir() and _extract_version(p.name, "iteration_v") >= 0
    ]
    if not iteration_dirs:
        raise FileNotFoundError(f"{latest_project} 下未找到 iteration_v* 目录")

    iteration_dirs.sort(key=lambda p: _extract_version(p.name, "iteration_v"), reverse=True)
    latest_iteration = iteration_dirs[0]

    kg_path = latest_iteration / "knowledge_graph.json"
    if not kg_path.exists():
        raise FileNotFoundError(f"未找到知识图谱文件: {kg_path}")

    return kg_path


def convert_kg(kg_path: Path, out_path: Path) -> None:
    """将 knowledge_graph.json (jsonl) 转为 server 侧使用的 data.json 结构。"""
    items = []
    with kg_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    nodes = []
    node_index = {}  # name -> idx
    links = []
    sents = []

    for item in items:
        sent = item.get("sentText", "").strip()
        if not sent:
            continue

        sent_idx = len(sents)
        sents.append(sent)

        for rel in item.get("relationMentions", []):
            h = rel.get("em1Text", "").strip()
            t = rel.get("em2Text", "").strip()
            r = rel.get("label", "").strip()
            if not h or not t or not r:
                continue

            # 建节点
            for name in (h, t):
                if name not in node_index:
                    idx = len(nodes)
                    node_index[name] = idx
                    nodes.append({
                        "id": str(idx),
                        "name": name,
                        "category": 0,
                        "draggable": True,
                        "value": 1,
                        "lines": [sent_idx],
                        "symbolSize": 20
                    })
                else:
                    idx = node_index[name]
                    if sent_idx not in nodes[idx]["lines"]:
                        nodes[idx]["lines"].append(sent_idx)
                    nodes[idx]["value"] += 1

            source_idx = node_index[h]
            target_idx = node_index[t]
            links.append({
                "source": source_idx,
                "target": target_idx,
                "name": r,
                "sent": sent_idx
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "nodes": nodes,
            "links": links,
            "sents": sents
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved graph data to {out_path} (nodes={len(nodes)}, links={len(links)}, sents={len(sents)})")


def main():
    kg_path = find_latest_kg_path()
    out_path = Path("server/data/data.json")
    print(f"Using KG file: {kg_path}")
    convert_kg(kg_path, out_path)


if __name__ == "__main__":
    main()

