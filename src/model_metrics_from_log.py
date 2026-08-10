import re
import pickle
from pathlib import Path
from collections import defaultdict

log_path = Path("/Odyssey/private/o23gauvr/code/FASCINATION/logs/job_41816.log")
out_path = Path("/Odyssey/private/o23gauvr/code/FASCINATION/pickle/model_metrics_article_long_new_no_filt.pkl")

ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
proc_re = re.compile(r"^\[(\d+)/(\d+)\]\s+Processing:\s+([A-Za-z0-9_]+)")
all_re = re.compile(r"^ALL\s+\(n=(\d+)\s+samples\):")
metric_re = re.compile(r"^\s*([a-z0-9_]+)\.+\s+([-+0-9.eE]+)\s*$")
date_re = re.compile(r"\d{8}_\d{6}")
cr_re = re.compile(r"CR_([0-9]+(?:\.[0-9]+)?)")

text = log_path.read_text(encoding="utf-8", errors="ignore")
lines = [ansi_re.sub("", ln) for ln in text.splitlines()]

# 1) Collect checkpoint paths from the header list
checkpoint_paths = []
for ln in lines:
    s = ln.strip()
    if s.startswith("- /") and s.endswith(".pth.tar"):
        checkpoint_paths.append(s[2:].strip())

if not checkpoint_paths:
    raise RuntimeError("No checkpoint paths found in log header.")

# 2) Parse each [i/15] block and its ALL metrics table
parsed = []  # entries: (idx, ckpt_name_from_processing, n_samples, metrics_dict)
current_idx = None
current_ckpt_name = None
i = 0
while i < len(lines):
    ln = lines[i].strip()

    m_proc = proc_re.match(ln)
    if m_proc:
        current_idx = int(m_proc.group(1)) - 1  # 0-based
        current_ckpt_name = m_proc.group(3)

    m_all = all_re.match(ln)
    if m_all and current_idx is not None:
        n_samples = int(m_all.group(1))
        metrics = {}
        j = i + 1
        while j < len(lines):
            mj = metric_re.match(lines[j])
            if mj:
                key = mj.group(1)
                val = float(mj.group(2))
                metrics[key] = val
                j += 1
                continue
            # stop on next section delimiter or blank gap after starting metrics
            if metrics and (lines[j].strip().startswith("===") or lines[j].strip().startswith("[")):
                break
            j += 1

        if metrics:
            parsed.append((current_idx, current_ckpt_name, n_samples, metrics))
        i = j
        continue

    i += 1

if not parsed:
    raise RuntimeError("No seasonal metric summaries parsed from log.")

# 3) Rebuild results structure compatible with compute_metrics_for_checkpoints output:
# results[model_name][cr]["all"] = {"SSP": metrics_plus_meta, "GRAD": {...}}
results = defaultdict(dict)

for idx, ckpt_name_proc, n_samples, metrics in parsed:
    if idx < 0 or idx >= len(checkpoint_paths):
        continue
    ckpt_path = checkpoint_paths[idx]
    ckpt = Path(ckpt_path)

    # Model name logic mirrors test_metrics.py
    path_parts = ckpt_path.split("/")
    date_match = next((p for p in path_parts if date_re.fullmatch(p)), None)
    ckpt_name = ckpt.stem.split(".")[0]
    model_name = f"MLIC_{date_match}_{ckpt_name}" if date_match else f"MLIC_UNKNOWN_{ckpt_name}"

    cr_match = cr_re.search(ckpt_path)
    cr = float(cr_match.group(1)) if cr_match else float("nan")

    ssp_metrics = dict(metrics)
    ssp_metrics["model_type"] = "MLIC"
    ssp_metrics["checkpoint_path"] = ckpt_path
    ssp_metrics["season"] = "all"
    ssp_metrics["num_samples"] = n_samples

    grad_metrics = {
        "season": "all",
        "num_samples": n_samples,
    }

    if cr not in results[model_name]:
        results[model_name][cr] = {}
    results[model_name][cr]["all"] = {"SSP": ssp_metrics, "GRAD": grad_metrics}

# 4) Save
results = dict(results)
with out_path.open("wb") as f:
    pickle.dump(results, f)

print(f"Saved: {out_path}")
print(f"Models: {len(results)}")
print(f"Entries parsed: {len(parsed)}")
print(f"Results: {results}")