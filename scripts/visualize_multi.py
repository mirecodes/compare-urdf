"""
visualize_multi.py
------------------
run_multi_compare.py 가 생성한 multi_report.yml 을 읽어
strip plot 3개를 생성한다.

  - 각 점(dot) = (object, joint) 쌍
  - x축 = predictor method
  - 3개 plot:
      out/multi_type_match.png
      out/multi_origin_dist.png
      out/multi_axis_angle.png

사용법:
  python scripts/visualize_multi.py
  python scripts/visualize_multi.py --report out/multi_report.yml
  python scripts/visualize_multi.py --report out/multi_report.yml --out-dir out/
"""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def parse_value(val) -> tuple[float, bool]:
    """
    숫자 또는 괄호로 감싼 문자열 → (float, is_prismatic).
    prismatic 은 괄호 표기로 판단.
    """
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


# ---------------------------------------------------------------------------
# 데이터 로딩 → long-form DataFrame
# ---------------------------------------------------------------------------

def build_dataframe(summary: dict, methods: list[str]) -> pd.DataFrame:
    """
    summary 딕셔너리를 long-form DataFrame 으로 변환한다.

    컬럼: Method, Object, JointGT, Value, Metric, Is_Prismatic
    """
    rows = []

    for obj_name, pred_dict in summary.items():
        for method in methods:
            joint_rows = pred_dict.get(method, [])
            if not joint_rows:
                continue
            for jrow in joint_rows:
                if not isinstance(jrow, dict):
                    continue

                gt_joint = jrow.get("joint_gt", "?")
                type_match = bool(jrow.get("type_match", False))

                # 1. Type Match
                rows.append({
                    "Method": method,
                    "Object": obj_name,
                    "JointGT": gt_joint,
                    "Metric": "Type Match Rate",
                    "Value": 1 if type_match else 0,
                    "Is_Prismatic": False,
                })

                # 2. Origin dist (type_match 여부와 무관하게 표시)
                origin_val, is_prismatic = parse_value(jrow.get("origin_dist_m", np.nan))
                rows.append({
                    "Method": method,
                    "Object": obj_name,
                    "JointGT": gt_joint,
                    "Metric": "Joint Origin Error (meters)",
                    "Value": origin_val,
                    "Is_Prismatic": is_prismatic,
                })

                # 3. Axis angle
                axis_val, _ = parse_value(jrow.get("axis_angle_deg", np.nan))
                rows.append({
                    "Method": method,
                    "Object": obj_name,
                    "JointGT": gt_joint,
                    "Metric": "Joint Axis Error (degrees)",
                    "Value": axis_val,
                    "Is_Prismatic": False,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_metric(
    df: pd.DataFrame,
    metric_name: str,
    out_path: Path,
    title: str,
    ylabel: str,
    order: list[str],
    palette: str = "Set2",
) -> None:
    """
    단일 metric 에 대한 strip plot 을 그리고 저장한다.
    origin dist 는 prismatic(hollow dot)과 revolute(solid dot) 를 구분한다.
    """
    subset = df[df["Metric"] == metric_name].copy()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    set2_colors = sns.color_palette(palette, len(order))
    method_color = {m: set2_colors[i] for i, m in enumerate(order)}

    rng = np.random.default_rng(seed=42)

    if metric_name == "Joint Origin Error (meters)":
        normal = subset[~subset["Is_Prismatic"]]
        prismatic = subset[subset["Is_Prismatic"]]

        if not normal.empty:
            sns.stripplot(
                data=normal, x="Method", y="Value",
                order=order, size=11, jitter=0.2, alpha=0.8,
                palette=palette, hue="Method", legend=False, ax=ax,
            )

        if not prismatic.empty:
            for i, method in enumerate(order):
                pts = prismatic[prismatic["Method"] == method]["Value"].dropna()
                if pts.empty:
                    continue
                jitter = rng.uniform(-0.2, 0.2, size=len(pts))
                ax.scatter(
                    x=i + jitter, y=pts.values,
                    s=11 ** 2,
                    facecolors="none",
                    edgecolors=method_color[method],
                    linewidths=2,
                    zorder=3,
                )

        # 평균선: revolute 만 사용
        medians = normal.groupby("Method", observed=True)["Value"].mean().reindex(order)

    else:
        if not subset.dropna(subset=["Value"]).empty:
            sns.stripplot(
                data=subset, x="Method", y="Value",
                order=order, size=11, jitter=0.2, alpha=0.8,
                palette=palette, hue="Method", legend=False, ax=ax,
            )
        medians = subset.groupby("Method", observed=True)["Value"].mean().reindex(order)

    # 평균 가로선 + 텍스트
    for i, method in enumerate(order):
        m_val = medians.get(method, np.nan) if hasattr(medians, "get") else medians.iloc[i] if i < len(medians) else np.nan
        if isinstance(m_val, float) and not np.isnan(m_val):
            ax.plot(
                [i - 0.25, i + 0.25], [m_val, m_val],
                color="#e74c3c", lw=5, solid_capstyle="round", zorder=5,
            )
            ax.text(
                i, m_val, f"{m_val:.3f}",
                color="black", ha="center", va="bottom",
                fontsize=10, fontweight="bold", zorder=6,
            )

    # Type Match 전용 y 범위
    if metric_name == "Type Match Rate":
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Failure (0)", "Success (1)"])

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Method", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  [저장] {out_path}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-joint URDF 비교 결과 시각화")
    parser.add_argument(
        "--report", "-r",
        default=str(ROOT / "out" / "multi_report.yml"),
        help="리포트 YAML 경로 (기본: out/multi_report.yml)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "out"),
        help="출력 디렉토리 (기본: out/)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="표시할 method 목록 (기본: 리포트에서 자동 감지)",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[ERROR] 리포트 없음: {report_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with report_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    summary = data.get("summary", {})
    if not summary:
        print("[ERROR] 리포트에 summary 가 없습니다.")
        sys.exit(1)

    # method 목록 자동 감지 (또는 사용자 지정)
    if args.methods:
        methods = args.methods
    else:
        methods_set: set[str] = set()
        for pred_dict in summary.values():
            if isinstance(pred_dict, dict):
                methods_set.update(pred_dict.keys())
        methods = sorted(methods_set)

    print(f"\n  Detected methods: {methods}")
    print(f"  Objects: {list(summary.keys())}\n")

    df = build_dataframe(summary, methods)

    sns.set_theme(style="whitegrid")

    plot_configs = [
        {
            "metric": "Type Match Rate",
            "filename": "multi_type_match.png",
            "title": "Joint Type Match Rate",
            "ylabel": "Success (1) / Failure (0)",
        },
        {
            "metric": "Joint Origin Error (meters)",
            "filename": "multi_origin_dist.png",
            "title": "Joint Origin Position Error",
            "ylabel": "Error (meters)",
        },
        {
            "metric": "Joint Axis Error (degrees)",
            "filename": "multi_axis_angle.png",
            "title": "Joint Axis Orientation Error",
            "ylabel": "Error (degrees)",
        },
    ]

    for cfg in plot_configs:
        plot_metric(
            df=df,
            metric_name=cfg["metric"],
            out_path=out_dir / cfg["filename"],
            title=cfg["title"],
            ylabel=cfg["ylabel"],
            order=methods,
        )

    print("\n  완료.")


if __name__ == "__main__":
    main()
