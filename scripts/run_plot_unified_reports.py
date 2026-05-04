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
sys.path.insert(0, str(ROOT / "utils"))

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

def build_dataframe_for_categories(reports: list[Path], methods: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    categories = []
    
    for report_path in reports:
        with report_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        summary = data.get("summary", {})
        if not summary:
            continue
            
        # Extract category from meta or fallback to parent directory name
        meta = data.get("meta", {})
        category = meta.get("category_filter", report_path.parent.name)
        if category not in categories:
            categories.append(category)

        for obj_name, obj_data in summary.items():
            gt_joint_count = obj_data.get("gt_joint_count", 0)
            predictors_data = obj_data.get("predictors", {})
            
            for method in methods:
                joint_rows = predictors_data.get(method, [])
                for jrow in joint_rows:
                    # 1. Type Match
                    rows.append({
                        "Category": category, "Method": method, "Object": obj_name, "Metric": "Joint Type Match Rate",
                        "Value": 1.0 if jrow.get("type_match", False) else 0.0, "Is_Prismatic": False
                    })
                    # 2. Origin Error
                    dist_raw = jrow.get("origin_dist_m")
                    dist, is_p = parse_value(dist_raw)
                    if not np.isnan(dist):
                        rows.append({
                            "Category": category, "Method": method, "Object": obj_name, "Metric": "Joint Origin Error (m)",
                            "Value": dist, "Is_Prismatic": is_p
                        })
                    # 3. Axis Error
                    angle_raw = jrow.get("axis_angle_deg")
                    angle, _ = parse_value(angle_raw)
                    if not np.isnan(angle):
                        rows.append({
                            "Category": category, "Method": method, "Object": obj_name, "Metric": "Joint Axis Error (deg)",
                            "Value": angle, "Is_Prismatic": False
                        })
                
                # 4. Error Score (Per Object)
                success_count = sum(1 for jrow in joint_rows if jrow.get("type_match", False))
                error_score = max(0, gt_joint_count - success_count)
                rows.append({
                    "Category": category, "Method": method, "Object": obj_name, "Metric": "Joint Reconstruction Error Score",
                    "Value": float(error_score), "Is_Prismatic": False
                })
                
    return pd.DataFrame(rows), categories

def plot_metric_multi_cat(df, metric_name, out_path, title, ylabel, methods_order, categories_order):
    subset = df[df["Metric"] == metric_name].copy()
    if subset.empty: return

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    
    # Use Set2 colors for categories
    cat_colors = sns.color_palette("Set2", n_colors=len(categories_order))
    cat_palette = dict(zip(categories_order, cat_colors))
    rng = np.random.default_rng(seed=42)

    if metric_name == "Joint Origin Error (m)":
        normal = subset[~subset["Is_Prismatic"]]
        prismatic = subset[subset["Is_Prismatic"]]
        if not normal.empty:
            sns.stripplot(data=normal, x="Method", y="Value", hue="Category", 
                          order=methods_order, hue_order=categories_order, 
                          size=11, jitter=0.2, dodge=False, alpha=0.8, palette=cat_palette, ax=ax)
        if not prismatic.empty:
            for i, method in enumerate(methods_order):
                for j, cat in enumerate(categories_order):
                    pts = prismatic[(prismatic["Method"] == method) & (prismatic["Category"] == cat)]["Value"].dropna()
                    if pts.empty: continue
                    jitter = rng.uniform(-0.2, 0.2, size=len(pts))
                    color = cat_palette.get(cat, "black")
                    ax.scatter(x=i + jitter, y=pts.values, s=11**2, facecolors="none", edgecolors=color, linewidths=2.5, zorder=3)
    else:
        sns.stripplot(data=subset, x="Method", y="Value", hue="Category", 
                      order=methods_order, hue_order=categories_order, 
                      size=11, jitter=0.2, dodge=False, alpha=0.8, palette=cat_palette, ax=ax)

    # Calculate overall means per method
    medians = subset.groupby("Method", observed=True)["Value"].mean()

    for i, method in enumerate(methods_order):
        m_val = medians.get(method, np.nan)
        if not np.isnan(m_val):
            ax.plot([i - 0.25, i + 0.25], [m_val, m_val], color="#e74c3c", lw=5, solid_capstyle="round", zorder=5)
            ax.text(i, m_val, f"{m_val:.3f}", color="black", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)

    if metric_name == "Joint Type Match Rate":
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Failure (0)", "Success (1)"])

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Method", fontsize=13)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    filtered_handles = [by_label[c] for c in categories_order if c in by_label]
    filtered_labels = [c for c in categories_order if c in by_label]
    ax.legend(filtered_handles, filtered_labels, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [저장] {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True, help="List of report.yml paths")
    parser.add_argument("--out_dir", required=True, help="Directory to save the merged plots")
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.reports]
    valid_reports = [p for p in report_paths if p.exists()]
    if not valid_reports:
        print("No valid reports found.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # X-axis order fixed
    methods = ["aa", "screw", "ours"]

    df, categories = build_dataframe_for_categories(valid_reports, methods)
    if df.empty:
        print("DataFrame is empty. Nothing to plot.")
        return

    print(f"\n  Plotting methods: {methods}")
    print(f"  Categories: {categories}\n")

    sns.set_theme(style="whitegrid")

    plot_configs = [
        {"metric": "Joint Type Match Rate", "filename": "multi_type_match.png", "title": "Joint Type Match Rate by Category", "ylabel": "Success (1) / Failure (0)"},
        {"metric": "Joint Origin Error (m)", "filename": "multi_origin_dist.png", "title": "Joint Origin Position Error by Category", "ylabel": "Error (m)"},
        {"metric": "Joint Axis Error (deg)", "filename": "multi_axis_angle.png", "title": "Joint Axis Orientation Error by Category", "ylabel": "Error (deg)"},
        {"metric": "Joint Reconstruction Error Score", "filename": "multi_error_score.png", "title": "Joint Reconstruction Error Score by Category", "ylabel": "Total Errors"},
    ]
    
    for cfg in plot_configs:
        out_path = out_dir / cfg["filename"]
        plot_metric_multi_cat(df, cfg["metric"], out_path, cfg["title"], cfg["ylabel"], methods, categories)

if __name__ == "__main__":
    main()
