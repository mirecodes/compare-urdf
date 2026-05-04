"""
autofill_mapping.py
-------------------
mapping.yml 에서 joint_pairs 가 없거나 null 인 predictor 를
GT ↔ pred joint 의 world-frame 위치 거리 기반으로 자동 매핑하여 채운다.

매핑 알고리즘:
  - GT 의 movable joint (전체) 와 pred 의 movable joint (전체) 의
    world-frame 원점 간 L2 거리 행렬을 계산한다.
  - scipy 의 Hungarian 알고리즘(linear_sum_assignment)으로
    전역 최적 1:1 매칭을 구한다.
  - GT joint 수가 pred joint 수보다 많은 경우, 매칭되지 않은 GT joint 는
    joint_pairs 에 포함되지 않는다 (null 유지).
  - 이미 joint_pairs 가 채워진 predictor 는 --overwrite 없이는 건드리지 않는다.

사용법:
  python scripts/autofill_mapping.py --input  cfgs/mapping_template.yml
                                     --output cfgs/mapping_filled.yml

  # 이미 채워진 joint_pairs 도 덮어쓰기:
  python scripts/autofill_mapping.py --input cfgs/mapping.yml \\
                                     --output cfgs/mapping_filled.yml \\
                                     --overwrite
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))

from compare_multi_joint import (  # noqa: E402
    make_z_rotation_matrix,
    MOVABLE_TYPES,
)
from compare_urdf import _normalize  # noqa: E402
import yourdfpy  # noqa: E402


# ---------------------------------------------------------------------------
# 모든 movable joint 로드 (필터 없이, z_rotation 적용)
# ---------------------------------------------------------------------------

def load_all_movable_joints(
    urdf_path: Path,
    z_rotation_deg: float = 0.0,
) -> Optional[dict[str, np.ndarray]]:
    """
    URDF 의 모든 movable joint 를 로드하고, z_rotation 을 적용한
    world-frame 원점 좌표(3D 벡터)를 반환한다.

    반환: {joint_name: np.ndarray([x, y, z])} 또는 로드 실패 시 None
    """
    try:
        model = yourdfpy.URDF.load(str(urdf_path))
    except Exception as e:
        print(f"    [ERROR] URDF 로드 실패 ({urdf_path.name}): {e}")
        return None

    R_z = make_z_rotation_matrix(z_rotation_deg)

    result: dict[str, np.ndarray] = {}
    for jname, joint in model.joint_map.items():
        if joint.type not in MOVABLE_TYPES:
            continue

        parent_link = joint.parent
        world_T_parent = model.get_transform(parent_link)
        world_T_joint = world_T_parent @ joint.origin
        world_T_joint_rot = R_z @ world_T_joint

        result[jname] = world_T_joint_rot[:3, 3].copy()

    return result


# ---------------------------------------------------------------------------
# Hungarian 매칭
# ---------------------------------------------------------------------------

def match_by_position(
    gt_joints: dict[str, np.ndarray],
    pred_joints: dict[str, np.ndarray],
) -> dict[str, str]:
    """
    GT joint 원점과 pred joint 원점 간 L2 거리를 기반으로
    Hungarian 알고리즘을 사용해 최적 1:1 매핑을 구한다.

    반환: {gt_joint_name: pred_joint_name}

    주의: GT joint 수 > pred joint 수 이면 일부 GT joint 는 매칭되지 않음.
          Pred joint 수 > GT joint 수 이면 일부 pred joint 는 사용되지 않음.
    """
    if not gt_joints or not pred_joints:
        return {}

    gt_names = list(gt_joints.keys())
    pred_names = list(pred_joints.keys())

    gt_pts = np.stack([gt_joints[n] for n in gt_names])       # (N_gt, 3)
    pred_pts = np.stack([pred_joints[n] for n in pred_names])  # (N_pred, 3)

    # 거리 행렬 (N_gt x N_pred)
    diff = gt_pts[:, None, :] - pred_pts[None, :, :]           # (N_gt, N_pred, 3)
    cost_matrix = np.linalg.norm(diff, axis=-1)                 # (N_gt, N_pred)

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # scipy 없을 경우 greedy fallback
        print("  [WARN] scipy 없음 → greedy 매칭으로 대체합니다.")
        row_ind, col_ind = _greedy_match(cost_matrix)

    pairs: dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        gt_name = gt_names[r]
        pred_name = pred_names[c]
        dist = cost_matrix[r, c]
        pairs[gt_name] = pred_name
        print(f"    {gt_name} ↔ {pred_name}  (dist={dist:.4f}m)")

    return pairs


def _greedy_match(cost_matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """Greedy 최근접 매칭 (scipy 없을 때 fallback)."""
    n_gt, n_pred = cost_matrix.shape
    used_pred: set[int] = set()
    row_ind, col_ind = [], []

    # 각 GT joint 에 대해 가장 가까운 미사용 pred joint 선택
    for r in range(n_gt):
        best_c = -1
        best_dist = float("inf")
        for c in range(n_pred):
            if c in used_pred:
                continue
            if cost_matrix[r, c] < best_dist:
                best_dist = cost_matrix[r, c]
                best_c = c
        if best_c >= 0:
            row_ind.append(r)
            col_ind.append(best_c)
            used_pred.add(best_c)

    return row_ind, col_ind


# ---------------------------------------------------------------------------
# mapping.yml 처리
# ---------------------------------------------------------------------------

def autofill_mapping(
    raw: dict,
    base_dir: Path,
    overwrite: bool = False,
) -> dict:
    """
    raw mapping dict 를 받아 joint_pairs 가 없는 predictor 를
    위치 기반 자동 매핑으로 채운다.
    """
    out = {}

    for obj_name, obj_data in raw.items():
        if not isinstance(obj_data, dict):
            out[obj_name] = obj_data
            continue

        gt_raw = obj_data.get("gt")
        if gt_raw is None:
            print(f"[WARN] '{obj_name}': gt 블록 없음. skip.")
            out[obj_name] = obj_data
            continue

        gt_path = (base_dir / gt_raw["urdf_path"]).resolve()
        if not gt_path.exists():
            print(f"[WARN] '{obj_name}': GT URDF 없음 → {gt_path}. skip.")
            out[obj_name] = obj_data
            continue

        gt_z = float(gt_raw.get("z_rotation", 0))

        print(f"\n{'─'*55}")
        print(f"  Object: {obj_name}")
        print(f"  GT:     {gt_path.name}  (z_rotation={gt_z}°)")
        print(f"{'─'*55}")

        # GT joints 로드 (전체)
        gt_joints_xyz = load_all_movable_joints(gt_path, gt_z)
        if not gt_joints_xyz:
            print(f"  [WARN] GT 에 movable joint 없음.")

        obj_out: dict = {"gt": obj_data["gt"]}

        for key, val in obj_data.items():
            if key == "gt":
                continue
            if not isinstance(val, dict):
                obj_out[key] = val
                continue

            # 이미 joint_pairs 가 채워진 경우
            existing_pairs = val.get("joint_pairs")
            if existing_pairs and not overwrite:
                print(f"  [{key}] joint_pairs 이미 존재 → 유지 (--overwrite 로 덮어쓰기 가능)")
                obj_out[key] = val
                continue

            pred_path_raw = val.get("urdf_path")
            if pred_path_raw is None:
                obj_out[key] = val
                continue

            pred_path = (base_dir / pred_path_raw).resolve()
            if not pred_path.exists():
                print(f"  [{key}] URDF 없음 → {pred_path}. joint_pairs=null 유지.")
                obj_out[key] = val
                continue

            pred_z = float(val.get("z_rotation", 0))
            print(f"  [{key}] {pred_path.name}  (z_rotation={pred_z}°)")

            # Pred joints 로드
            pred_joints_xyz = load_all_movable_joints(pred_path, pred_z)
            if pred_joints_xyz is None:
                print(f"    [WARN] pred URDF 로드 실패. joint_pairs=null 유지.")
                obj_out[key] = val
                continue
            
            if not pred_joints_xyz:
                print(f"    [WARN] pred 에 movable joint 없음. joint_pairs=null 유지.")
                obj_out[key] = val
                continue

            print(f"    GT joints  : {list(gt_joints_xyz.keys())}")
            print(f"    Pred joints: {list(pred_joints_xyz.keys())}")

            # 위치 기반 매핑
            pairs = match_by_position(gt_joints_xyz, pred_joints_xyz)

            new_val = dict(val)
            new_val["joint_pairs"] = pairs if pairs else None
            obj_out[key] = new_val

        out[obj_name] = obj_out

    return out


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="위치 기반 joint 자동 매핑 — mapping.yml 의 joint_pairs 를 채운다."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="입력 mapping YAML 경로 (joint_pairs 가 없는 항목을 채움)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="출력 mapping YAML 경로",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 joint_pairs 가 있는 항목도 재매핑",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        print(f"[ERROR] 입력 파일 없음: {in_path}")
        sys.exit(1)

    with in_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        print("[ERROR] 입력 파일이 비어있습니다.")
        sys.exit(1)

    print(f"\n{'═'*55}")
    print(f"  Autofill Mapping")
    print(f"  input : {in_path}")
    print(f"  output: {out_path}")
    print(f"  overwrite: {args.overwrite}")
    print(f"{'═'*55}")

    filled = autofill_mapping(raw, base_dir=in_path.parent, overwrite=args.overwrite)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(
            filled,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
            default_style=None,
        )

    print(f"\n  [저장] {out_path}")


if __name__ == "__main__":
    main()
