"""Draw accuracy / latency / size charts from results/results.json."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/results.json")
res = json.loads(path.read_text())["variants"]
names = list(res)
charts = [("accuracy", "Accuracy (exact match)"), ("latency_ms_p50", "p50 latency, batch 1 (ms)"),
          ("weights_mb", "Weights size (MB)")]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (key, title) in zip(axes, charts, strict=True):
    vals = [res[n][key] for n in names]
    ax.bar(range(len(names)), vals, color="#4c72b0")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}" if key == "accuracy" else f"{v:.0f}", ha="center", va="bottom")
fig.tight_layout()
out = path.parent / "comparison.png"
fig.savefig(out, dpi=110)
print(out)
