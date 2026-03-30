"""
run_multi_compare.py
--------------------
mapping.yml 을 읽어 여러 object/joint 를 일괄 비교하고 YAML 리포트를 out/ 에 저장한다.

사용법:
  python scripts/run_multi_compare.py --mapping cfgs/mapping.yml
  python scripts/run_multi_compare.py --mapping cfgs/mapping.yml --no-save

리포트 구조 (out/multi_report.yml):
  meta:
    generated_at: ...
    mapping: cfgs/mapping.yml
  summary:
    {object}:
      {predictor}:
        - joint_gt: ...
          joint_pred: ...
          type_match: bool
          origin_dist_m: float
          axis_angle_deg: float
  details:
    {object}:
      {predictor}:
        - joint_name_gt: ...
          joint_name_pred: ...
          type_gt: ...
          type_pred: ...
          origin_gt_xyz: [x, y, z]
          origin_pred_xyz: [x, y, z]
          axis_gt_xyz: [x, y, z]
          axis_pred_xyz: [x, y, z]
          axis_dot_abs: float
          notes: [...]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_urdf import (  # noqa: E402
    result_to_summary_dict,
    result_to_detail_dict,
)
from compare_multi_joint import (  # noqa: E402
    parse_mapping_yml,
    compare_by_mapping,
)


# ---------------------------------------------------------------------------
# 결과 직렬화
# ---------------------------------------------------------------------------

def _result_to_summary_row(r) -> dict:
    """ComparisonResult → summary 행 dict."""
    row = {
        "joint_gt": r.joint_name_gt,
        "joint_pred": r.joint_name_pred,
    }
    row.update(result_to_summary_dict(r))
    return row


def _result_to_detail_row(r) -> dict:
    """ComparisonResult → detail 행 dict."""
    return result_to_detail_dict(r)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-joint URDF 비교 — mapping.yml 기반")
    parser.add_argument(
        "--mapping", "-m",
        default=str(ROOT / "cfgs" / "mapping.yml"),
        help="mapping 파일 경로 (기본: cfgs/mapping.yml)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과를 파일에 저장하지 않음 (터미널 출력만)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "out" / "multi_report.yml"),
        help="출력 YAML 경로 (기본: out/multi_report.yml)",
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"[ERROR] mapping 파일 없음: {mapping_path}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Multi-Joint URDF 비교")
    print(f"  mapping: {mapping_path}")
    print(f"{'═'*60}\n")

    # mapping.yml 파싱
    object_mappings = parse_mapping_yml(mapping_path)

    if not object_mappings:
        print("[ERROR] 유효한 object mapping 이 없습니다.")
        sys.exit(1)

    print(f"총 {len(object_mappings)} 개 object 발견.\n")

    summary_section: dict = {}
    detail_section: dict = {}

    for obj_map in object_mappings:
        obj_name = obj_map.name
        pred_names = list(obj_map.predictors.keys())
        print(f"{'─'*60}")
        print(f"  Object: {obj_name}  |  predictors: {pred_names}")
        print(f"{'─'*60}")

        # 비교 수행
        results = compare_by_mapping(obj_map)   # {pred_name: [ComparisonResult]}

        obj_summary: dict = {}
        obj_detail: dict = {}

        for pred_name, pair_results in results.items():
            obj_summary[pred_name] = [_result_to_summary_row(r) for r in pair_results]
            obj_detail[pred_name]  = [_result_to_detail_row(r)  for r in pair_results]

        summary_section[obj_name] = obj_summary
        detail_section[obj_name]  = obj_detail

    if args.no_save:
        print("\n[INFO] --no-save 지정: 파일 저장 생략.")
        return

    # YAML 리포트 저장
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mapping": str(mapping_path),
        },
        "summary": summary_section,
        "details": detail_section,
    }

    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n  [저장] {out_path}")


if __name__ == "__main__":
    main()
