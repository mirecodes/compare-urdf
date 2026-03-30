"""
run_compare.py
--------------
cfgs/config.yml 을 읽어 여러 카테고리/오브젝트를 일괄 비교하고
YAML 리포트를 out/ 에 저장한다.

사용법:
  python scripts/run_compare.py                        # cfgs/config.yml 기본
  python scripts/run_compare.py --config cfgs/other.yml
  python scripts/run_compare.py --no-save              # 터미널 출력만
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml  # pyyaml
import yourdfpy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_urdf import (  # noqa: E402
    compare_urdf_files,
    result_to_summary_dict,
    result_to_detail_dict,
    print_result,
    MOVABLE_TYPES,
)

PREDICTORS = ["ours", "aa", "screw"]  # GT 와 비교할 대상 폴더 목록


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def find_urdf(folder: Path, obj_name: str) -> Path:
    """
    {folder}/{obj_name}.urdf 를 찾는다.
    없으면 폴더 안의 첫 번째 .urdf 를 fallback 으로 사용한다.
    """
    exact = folder / f"{obj_name}.urdf"
    if exact.exists():
        return exact

    urdfs = sorted(folder.glob("*.urdf"))
    if not urdfs:
        raise FileNotFoundError(f"No .urdf file found in {folder}")
    print(f"[WARN] {exact} 없음 → fallback: {urdfs[0].name}")
    return urdfs[0]


# ---------------------------------------------------------------------------
# 단일 오브젝트 비교
# ---------------------------------------------------------------------------

def compare_object(
    category: str,
    obj: str,
    print_detail: bool = True,
) -> dict:
    """
    GT vs {ours, aa} 비교를 수행하고 리포트용 dict 를 반환한다.
    """
    base = ROOT / "data" / category   # data/{category}/

    gt_dir = base / "gt"
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT 폴더 없음: {gt_dir}")

    gt_urdf = find_urdf(gt_dir, obj)
    
    # GT movable joint 총 개수 파악
    gt_joint_count = 0
    try:
        model = yourdfpy.URDF.load(str(gt_urdf))
        gt_joint_count = sum(1 for j in model.joint_map.values() if j.type in MOVABLE_TYPES)
    except Exception as e:
        print(f"  [WARN] GT URDF 로드 실패 (개수 파악 불가): {e}")

    print(f"\n  ▷ {category}/{obj}  |  GT: {gt_urdf.name} (joints={gt_joint_count})")

    summary_block: dict = {}
    detail_block: dict = {}

    for pred in PREDICTORS:
        pred_dir = base / pred
        if not pred_dir.exists():
            continue

        try:
            pred_urdf = find_urdf(pred_dir, obj)
        except FileNotFoundError:
            continue

        result = compare_urdf_files(gt_urdf, pred_urdf)

        if print_detail:
            print_result(pred, result)

        summary_block[pred] = result_to_summary_dict(result)
        detail_block[pred]  = result_to_detail_dict(result)

    return {
        "object": obj,
        "gt_joint_count": gt_joint_count,
        "summary": summary_block,
        "details": detail_block,
    }


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="URDF 비교 — config 기반 배치 실행")
    parser.add_argument(
        "--config", "-c",
        default=str(ROOT / "cfgs" / "config.yml"),
        help="설정 파일 경로 (기본: cfgs/config.yml)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 저장 안 함 (터미널 출력만)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] config 파일 없음: {config_path}")
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    categories = cfg.get("categories", [])
    if not categories:
        print("[ERROR] config 에 categories 가 없습니다.")
        sys.exit(1)

    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)

    for cat_cfg in categories:
        category: str = cat_cfg["name"]
        objects: list[str] = cat_cfg.get("objects", [])

        if not objects:
            print(f"[SKIP] {category}: objects 목록이 비어있습니다.")
            continue

        print(f"\n{'═'*55}")
        print(f"  Category: {category}  ({len(objects)} objects)")
        print(f"{'═'*55}")

        # 카테고리 단위 리포트 조합
        report_objects: list[dict] = []
        for obj in objects:
            try:
                obj_report = compare_object(category, obj)
            except FileNotFoundError as e:
                print(f"  [ERROR] {obj}: {e}")
                continue
            report_objects.append(obj_report)

        if args.no_save:
            continue

        summary_section: dict = {}
        detail_section: dict = {}
        for obj_data in report_objects:
            obj = obj_data["object"]
            summary_section[obj] = {
                "gt_joint_count": obj_data["gt_joint_count"],
                "predictors": obj_data["summary"]
            }
            detail_section[obj] = obj_data["details"]

        report = {
            "meta": {
                "category": category,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path),
                "predictors": PREDICTORS,
            },
            "summary": summary_section,
            "details": detail_section,
        }

        # 카테고리별 출력 경로 (run_multi_compare 와 동일 패턴)
        out_path = out_dir / category / "report.yml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(
                report,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        print(f"\n  [저장] {out_path}")


if __name__ == "__main__":
    main()
