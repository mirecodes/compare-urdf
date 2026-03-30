"""
run_compare.py
--------------
cfgs/config.yml 을 읽어 여러 카테고리/오브젝트를 일괄 비교하고
YAML 리포트를 out/ 에 저장한다.

사용법:
  python scripts/run_compare.py                        # cfgs/config.yml 기본
  python scripts/run_compare.py --config cfgs/other.yml
  python scripts/run_compare.py --no-save              # 터미널 출력만

리포트 구조 (out/{category}_report.yml):
  [상단] summary  — type_match / origin_l2_m / axis_angle_deg
  [하단] details  — joint 이름, xyz 좌표, dot 등 전체 정보
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import yaml  # pyyaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_urdf import (  # noqa: E402
    compare_urdf_files,
    result_to_summary_dict,
    result_to_detail_dict,
    print_result,
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

    디렉토리 구조:
      data/{category}/gt/{obj}.urdf
      data/{category}/ours/{obj}.urdf
      data/{category}/aa/{obj}.urdf

    반환 구조:
      {
        "object": obj,
        "summary": { "ours": {...}, "aa": {...} },
        "details": { "ours": {...}, "aa": {...} },
      }
    """
    base = ROOT / "data" / category   # data/{category}/

    gt_dir = base / "gt"
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT 폴더 없음: {gt_dir}")

    gt_urdf = find_urdf(gt_dir, obj)
    print(f"\n  ▷ {category}/{obj}  |  GT: {gt_urdf.name}")

    summary_block: dict = {}
    detail_block: dict = {}

    for pred in PREDICTORS:
        pred_dir = base / pred
        if not pred_dir.exists():
            print(f"  [SKIP] {pred}/ 폴더 없음: {pred_dir}")
            continue

        try:
            pred_urdf = find_urdf(pred_dir, obj)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        result = compare_urdf_files(gt_urdf, pred_urdf)

        if print_detail:
            print_result(pred, result)

        summary_block[pred] = result_to_summary_dict(result)
        detail_block[pred]  = result_to_detail_dict(result)

    return {
        "object": obj,
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

        # ---- YAML 리포트 작성 ----
        # 상단: summary (카테고리 전체 + 오브젝트별)
        # 하단: details (오브젝트별 상세)

        summary_section: dict = {}
        detail_section: dict = {}
        for obj_data in report_objects:
            obj      = obj_data["object"]
            summary_section[obj] = obj_data["summary"]
            detail_section[obj]  = obj_data["details"]

        report = {
            # ── 상단 요약 ──
            "meta": {
                "category": category,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path),
                "predictors": PREDICTORS,
            },
            "summary": summary_section,
            # ── 하단 상세 ──
            "details": detail_section,
        }

        out_path = out_dir / f"{category}_report.yml"
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
