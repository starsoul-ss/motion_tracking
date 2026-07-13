#!/usr/bin/env python3
"""Copy a motion memmap while removing selected local motion IDs."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active_adaptation.utils.motion import MotionMinimalData, _write_motion_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--remove-ids", required=True, help="JSON list or file containing local motion IDs")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw = Path(args.remove_ids)
    remove = set(map(int, json.loads(raw.read_text() if raw.is_file() else args.remove_ids)))

    meta = json.loads((source / "meta_motion.json").read_text())
    labels = json.loads((source / "id_label.json").read_text())
    data = MotionMinimalData.load_memmap(str(source / "_tensordict"))
    keep = [i for i in range(len(labels)) if i not in remove]
    if remove - set(range(len(labels))):
        raise ValueError(f"remove IDs out of range: {sorted(remove - set(range(len(labels))))[:10]}")

    starts, ends = meta["starts"], meta["ends"]
    metadata_rows = [
        {key: values[i] for key, values in meta.get("info", {}).items()}
        for i in keep
    ]
    _write_motion_dataset(
        joint_names=meta["joint_names"],
        body_names=meta.get("body_names", []),
        metadata_rows=metadata_rows,
        id_labels=[labels[i] for i in keep],
        lengths=[ends[i] - starts[i] for i in keep],
        root_pos_chunks=[data.root_pos_w[starts[i]:ends[i]] for i in keep],
        root_quat_chunks=[data.root_quat_w[starts[i]:ends[i]] for i in keep],
        joint_pos_chunks=[data.joint_pos[starts[i]:ends[i]] for i in keep],
        mem_path=str(output),
    )
    print(json.dumps({"source": str(source), "output": str(output), "before": len(labels), "removed": len(remove), "after": len(keep)}))


if __name__ == "__main__":
    main()
