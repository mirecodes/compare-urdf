"""
visualize_multi.py
------------------
run_multi_compare.py 가 생성한 multi_report.yml 을 읽어 strip plot 4개를 생성한다.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def parse_value(val) -> tuple[float, bool]:
    if val is None:
        return np.nan, False
    if isinstance(val, str):
        is_prismatic = val.strip().startswith("(") and val.strip().endswith(")")
        clean = re.sub(r"[()]", "", val).strip()
        try:
            return float(clean), is_prismatic
        except ValueError:
            return np.nan, False
    try:
        return float(val), False
    except (TypeError, ValueError):
        return np.nan, False


def build_dataframe(summary: dict, methods: list[str]) -> pd.DataFrame:
    rows = []
    for obj_name, obj_data in summary.items():
        # 요약 구조: { gt_joint_count: N, predictors: { p_name: [results] } }
        gt_joint_count = obj_data.get("gt_joint_count", 0)
        predictors_data = obj_data.get("predictors", {})
        
        for method in methods:
            joint_rows = predictors_data.get(method, [])
            for jrow in joint_rows:
                # 1. Type Match
                rows.append({
                    "Method": method, "Object": obj_name, "Metric": "Joint Type Match Rate",
                    "Value": 1.0 if jrow.get("type_match", False) else 0.0, "Is_Prismatic": False
                })
                # 2. Origin Error
                dist_raw = jrow.get("origin_dist_m")
                dist, is_p = parse_value(dist_raw)
                if not np.isnan(dist):
                    rows.append({
                        "Method": method, "Object": obj_name, "Metric": "Joint Origin Error (m)",
                        "Value": dist, "Is_Prismatic": is_p
                    })
                # 3. Axis Error
                angle_raw = jrow.get("axis_angle_deg")
                angle, _ = parse_value(angle_raw)
                if not np.isnan(angle):
                    rows.append({
                        "Method": method, "Object": obj_name, "Metric": "Joint Axis Error (deg)",
                        "Value": angle, "Is_Prismatic": False
                    })
            
            # 4. Error Score (Per Object)
            success_count = sum(1 for jrow in joint_rows if jrow.get("type_match", False))
            error_score = max(0, gt_joint_count - success_count)
            rows.append({
                "Method": method, "Object": obj_name, "Metric": "Joint Reconstruction Error Score",
                "Value": float(error_score), "Is_Prismatic": False
            })
    return pd.DataFrame(rows)


def plot_metric(df, metric_name, out_path, title, ylabel, order):
    subset = df[df["Metric"] == metric_name].copy()
    if subset.empty: return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    palette = "Set2"
    set2_colors = sns.color_palette(palette, len(order))
    method_color = {m: set2_colors[i] for i, m in enumerate(order)}
    rng = np.random.default_rng(seed=42)

    if metric_name == "Joint Origin Error (m)":
        normal = subset[~subset["Is_Prismatic"]]
        prismatic = subset[subset["Is_Prismatic"]]
        if not normal.empty:
            sns.stripplot(data=normal, x="Method", y="Value", order=order, size=11, jitter=0.2, alpha=0.8, palette=palette, hue="Method", legend=False, ax=ax)
        if not prismatic.empty:
            for i, method in enumerate(order):
                pts = prismatic[prismatic["Method"] == method]["Value"].dropna()
                if pts.empty: continue
                jitter = rng.uniform(-0.2, 0.2, size=len(pts))
                ax.scatter(x=i + jitter, y=pts.values, s=11**2, facecolors="none", edgecolors=method_color[method], linewidths=2, zorder=3)
        medians = normal.groupby("Method", observed=True)["Value"].mean().reindex(order)
    else:
        sns.stripplot(data=subset, x="Method", y="Value", order=order, size=11, jitter=0.2, alpha=0.8, palette=palette, hue="Method", legend=False, ax=ax)
        medians = subset.groupby("Method", observed=True)["Value"].mean().reindex(order)

    for i, method in enumerate(order):
        m_val = medians.get(method, np.nan)
        if not np.isnan(m_val):
            ax.plot([i - 0.25, i + 0.25], [m_val, m_val], color="#e74c3c", lw=5, solid_capstyle="round", zorder=5)
            ax.text(i, m_val, f"{m_val:.3f}", color="black", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)

    if metric_name == "Joint Type Match Rate":
        ax.set_ylim(-0.2, 1.2); ax.set_yticks([0, 1]); ax.set_yticklabels(["Failure (0)", "Success (1)"])

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=13); ax.set_xlabel("Method", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [저장] {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="report.yml 경로")
    args = parser.parse_args()
    report_path = Path(args.report)
    if not report_path.exists(): return

    with report_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    summary = data.get("summary", {})
    if not summary: return

    # x-축 순서 고정 (AA -> Ours)
    methods = ["aa", "ours"]
    print(f"\n  Plotting methods: {methods}")
    print(f"  Objects: {list(summary.keys())}\n")

    df = build_dataframe(summary, methods)
    sns.set_theme(style="whitegrid")
    out_dir = report_path.parent
    plot_configs = [
        {"metric": "Joint Type Match Rate", "filename": "multi_type_match.png", "title": "Joint Type Match Rate", "ylabel": "Success (1) / Failure (0)"},
        {"metric": "Joint Origin Error (m)", "filename": "multi_origin_dist.png", "title": "Joint Origin Position Error", "ylabel": "Error (m)"},
        {"metric": "Joint Axis Error (deg)", "filename": "multi_axis_angle.png", "title": "Joint Axis Orientation Error", "ylabel": "Error (deg)"},
        {"metric": "Joint Reconstruction Error Score", "filename": "multi_error_score.png", "title": "Joint Reconstruction Error Score", "ylabel": "Total Errors"},
    ]
    for cfg in plot_configs:
        plot_metric(df, cfg["metric"], out_dir / cfg["filename"], cfg["title"], cfg["ylabel"], methods)

if __name__ == "__main__":
    main()
