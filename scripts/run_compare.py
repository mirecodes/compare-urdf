"""
run_multi_compare.py
-------------------
URDF 멀티 조인트 비교 실행 스크립트.
"""

import argparse
import sys
import math
from pathlib import Path
from datetime import datetime, timezone

import yaml

# 스크립트 경로 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))

from compare_urdf import ComparisonResult
from compare_multi_joint import parse_mapping_yml, compare_by_mapping


def _result_to_summary_row(r: ComparisonResult) -> dict:
    """YAML 상단 요약용 데이터 변환."""
    from compare_urdf import result_to_summary_dict
    return result_to_summary_dict(r)


def _result_to_detail_row(r: ComparisonResult) -> dict:
    """YAML 하단 상세용 데이터 변환."""
    from compare_urdf import result_to_detail_dict
    return result_to_detail_dict(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, help="YAML 매핑 파일 경로")
    parser.add_argument("--category", help="카테고리 이름 (출력 경로 자동 설정 및 필터링용)")
    parser.add_argument("--no-save", action="store_true", help="파일 저장 안함")
    parser.add_argument(
        "--out",
        help="출력 YAML 경로 (기본: out/multi_report.yml 또는 out/{category}/multi_report.yml)",
    )
    args = parser.parse_args()

    # 출력 경로 결정
    if args.out:
        out_path = Path(args.out)
    elif args.category:
        out_path = ROOT / "out" / args.category / "multi_report.yml"
    else:
        out_path = ROOT / "out" / "multi_report.yml"

    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"[ERROR] mapping 파일 없음: {mapping_path}")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  Multi-Joint URDF 비교")
    print(f"  mapping: {mapping_path}")
    if args.category:
        print(f"  filter : category='{args.category}'")
    print(f"{'═'*60}\n")

    # mapping.yml 파싱
    object_mappings = parse_mapping_yml(mapping_path)
    if not object_mappings:
        print("[ERROR] 유효한 object mapping 이 없습니다.")
        sys.exit(1)

    print(f"총 {len(object_mappings)} 개 object 발견.\n")

    summary_section: dict = {}
    detail_section: dict = {}

    processed_count = 0
    for obj_map in object_mappings:
        obj_name = obj_map.name
        
        # 카테고리 필터링: urdf_path 에 카테고리명이 포함되어 있는지 확인
        if args.category:
            if f"/{args.category}/" not in str(obj_map.gt.urdf_path):
                continue

        processed_count += 1
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

            # CLI 상세 출력
            for r in pair_results:
                icon = "✔" if r.type_match else "✘"
                dist_val = r.origin_dist_m
                dist_str = f"{dist_val:.4f}m" if not math.isnan(dist_val) else "N/A"
                if r.type_gt == "prismatic":
                    dist_str = f"({dist_str})"
                angle_str = f"{r.axis_angle_deg:.2f}°" if not math.isnan(r.axis_angle_deg) else "N/A"
                
                print(f"  [{pred_name}] {r.joint_name_gt} ↔ {r.joint_name_pred}  "
                      f"type={icon}  dist={dist_str:s}  angle={angle_str:s}")

        summary_section[obj_name] = {
            "gt_joint_count": obj_map.gt.joint_count,
            "predictors": obj_summary
        }
        detail_section[obj_name]  = obj_detail

    print(f"\n총 {processed_count}개 오브젝트가 필터링/처리되었습니다.")

    if processed_count == 0:
        print("[WARN] 처리된 오브젝트가 없습니다. 리포트를 생성하지 않습니다.")
        return

    if args.no_save:
        print("\n[INFO] --no-save 지정: 파일 저장 생략.")
        return

    # YAML 리포트 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mapping": str(mapping_path),
            "category_filter": args.category if args.category else "none",
        },
        "summary": summary_section,
        "details": detail_section,
    }

    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(report, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n  [저장] {out_path}")


if __name__ == "__main__":
    main()
