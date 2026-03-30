"""
compare_multi_joint.py
----------------------
mapping.yml 기반 멀티-joint URDF 비교 엔진.

기존 compare_urdf.py 의 JointInfo / ComparisonResult / 비교 함수들을 재사용하고,
아래를 추가로 제공한다:

  - make_z_rotation_matrix(angle_deg)  → 4×4 z축 회전 행렬
  - load_joints_filtered(...)          → 지정 joint 이름만 로드 + z_rotation 적용
  - compare_joint_pair(...)            → 단일 GT↔pred 비교 (compare_single_joint 위임)
  - compare_by_mapping(...)            → joint_pairs 전체를 순회하며 비교
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yourdfpy

# 기존 모듈 재사용
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare_urdf import (  # noqa: E402
    JointInfo,
    ComparisonResult,
    MOVABLE_TYPES,
    REVOLUTE_TYPES,
    _normalize,
    compare_single_joint,
)


# ---------------------------------------------------------------------------
# 데이터 구조체
# ---------------------------------------------------------------------------

@dataclass
class PredictorConfig:
    """mapping.yml 에서 파싱한 predictor 설정."""
    urdf_path: Path
    z_rotation_deg: float
    joint_pairs: dict[str, str]   # {gt_joint_name: pred_joint_name}


@dataclass
class GTConfig:
    """mapping.yml 에서 파싱한 GT 설정."""
    urdf_path: Path
    z_rotation_deg: float


@dataclass
class ObjectMapping:
    """object 하나에 대한 전체 매핑 설정."""
    name: str
    gt: GTConfig
    predictors: dict[str, PredictorConfig]   # {predictor_name: PredictorConfig}


# ---------------------------------------------------------------------------
# z축 회전
# ---------------------------------------------------------------------------

def make_z_rotation_matrix(angle_deg: float) -> np.ndarray:
    """
    z축 기준 회전 4×4 homogeneous transform 을 반환한다.
    angle_deg: 도(degree) 단위.
    """
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


# ---------------------------------------------------------------------------
# URDF 로딩 — 지정 joint 이름 필터 + z_rotation 적용
# ---------------------------------------------------------------------------

def load_joints_filtered(
    urdf_path: Path,
    joint_names: list[str],
    z_rotation_deg: float = 0.0,
) -> dict[str, JointInfo]:
    """
    URDF 를 로드하고 joint_names 에 해당하는 movable joint 만 반환한다.
    z_rotation_deg 만큼 base z축 회전을 적용하여 world-frame 좌표를 보정한다.

    반환: {joint_name: JointInfo}  (world-frame, z_rotation 적용 완료)
    """
    model = yourdfpy.URDF.load(str(urdf_path))
    R_z = make_z_rotation_matrix(z_rotation_deg)

    # URDF 내 movable joint 를 world-frame 으로 변환
    all_movable: dict[str, JointInfo] = {}
    for jname, joint in model.joint_map.items():
        if joint.type not in MOVABLE_TYPES:
            continue

        parent_link = joint.parent
        world_T_parent = model.get_transform(parent_link)
        world_T_joint = world_T_parent @ joint.origin

        # z_rotation 보정: 전체 object 좌표계에 R_z 를 추가로 적용
        world_T_joint_rot = R_z @ world_T_joint

        world_xyz = tuple(world_T_joint_rot[:3, 3])

        local_axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0])
        # axis 도 동일하게 회전 적용 (평행이동 없음 → 상단 3×3만 사용)
        world_axis = (R_z[:3, :3] @ world_T_joint[:3, :3]) @ local_axis
        world_axis = _normalize(tuple(world_axis))

        all_movable[jname] = JointInfo(
            name=jname,
            joint_type=joint.type,
            origin_xyz=world_xyz,
            origin_rpy=(0, 0, 0),
            axis_xyz=world_axis,
        )

    # 요청된 이름만 필터
    result: dict[str, JointInfo] = {}
    for name in joint_names:
        if name in all_movable:
            result[name] = all_movable[name]
        else:
            print(f"  [WARN] joint '{name}' not found in {urdf_path.name}. "
                  f"Available movable joints: {list(all_movable.keys())}")
    return result


# ---------------------------------------------------------------------------
# 단일 joint 쌍 비교 (compare_single_joint 위임)
# ---------------------------------------------------------------------------

def compare_joint_pair(gt: JointInfo, pred: JointInfo) -> ComparisonResult:
    """GT joint 하나와 pred joint 하나를 비교한다. (기존 로직 그대로 위임)"""
    return compare_single_joint(gt, pred)


# ---------------------------------------------------------------------------
# mapping 전체 비교
# ---------------------------------------------------------------------------

def _null_result(pred_name: str) -> ComparisonResult:
    """
    joint_pairs 매핑이 없는 predictor 에 대한 null 결과를 반환한다.
    - type_match = False  → Type Match 차트에 failure 로 카운트됨
    - origin_dist_m = NaN → Origin Error 평가에서 제외됨
    - axis_angle_deg = NaN → Axis Error 평가에서 제외됨
    """
    return ComparisonResult(
        joint_name_gt="none",
        joint_name_pred="none",
        type_gt="none",
        type_pred="none",
        type_match=False,
        origin_gt=(0.0, 0.0, 0.0),
        origin_pred=(0.0, 0.0, 0.0),
        origin_dist_m=float("nan"),
        axis_gt=(0.0, 0.0, 0.0),
        axis_pred=(0.0, 0.0, 0.0),
        axis_dot=float("nan"),
        axis_angle_deg=float("nan"),
        notes=[f"No joint_pairs mapping defined for predictor '{pred_name}'"],
    )


def compare_by_mapping(
    obj_mapping: ObjectMapping,
) -> dict[str, list[ComparisonResult]]:
    """
    ObjectMapping 에 정의된 모든 predictor 에 대해 joint_pairs 를 순회하며
    GT ↔ pred 비교를 수행한다.

    joint_pairs 가 없거나 null 인 predictor 는 null 결과(type_match=False, NaN 거리)
    1개를 생성한다 — Type Match 차트에 failure 로 카운트되며,
    Origin/Axis Error 평가에서는 제외된다.

    반환: {predictor_name: [ComparisonResult, ...]}
          joint_pairs 순서대로 정렬됨.
    """
    gt_cfg = obj_mapping.gt

    results: dict[str, list[ComparisonResult]] = {}

    for pred_name, pred_cfg in obj_mapping.predictors.items():
        # joint_pairs 가 없거나 null → null 결과 1개 생성
        if not pred_cfg.joint_pairs:
            print(f"  [{pred_name}] joint_pairs 없음 → null 결과 (type_match=False)")
            results[pred_name] = [_null_result(pred_name)]
            continue

        gt_joint_names = list(pred_cfg.joint_pairs.keys())
        pred_joint_names = list(pred_cfg.joint_pairs.values())

        # GT joints 로드
        gt_joints = load_joints_filtered(
            gt_cfg.urdf_path,
            gt_joint_names,
            gt_cfg.z_rotation_deg,
        )

        # Pred joints 로드
        pred_joints = load_joints_filtered(
            pred_cfg.urdf_path,
            pred_joint_names,
            pred_cfg.z_rotation_deg,
        )

        pair_results: list[ComparisonResult] = []
        for gt_name, pred_name_j in pred_cfg.joint_pairs.items():
            if gt_name not in gt_joints:
                print(f"  [SKIP] GT joint '{gt_name}' 를 찾을 수 없습니다. ({gt_cfg.urdf_path.name})")
                continue
            if pred_name_j not in pred_joints:
                print(f"  [SKIP] pred joint '{pred_name_j}' 를 찾을 수 없습니다. ({pred_cfg.urdf_path.name})")
                continue

            r = compare_joint_pair(gt_joints[gt_name], pred_joints[pred_name_j])
            pair_results.append(r)
            print(
                f"  [{pred_name}] {gt_name} ↔ {pred_name_j}"
                f"  type={'✔' if r.type_match else '✘'}"
                f"  dist={r.origin_dist_m:.4f}m"
                f"  angle={r.axis_angle_deg:.2f}°"
            )

        results[pred_name] = pair_results

    return results



# ---------------------------------------------------------------------------
# mapping.yml 파서
# ---------------------------------------------------------------------------

def parse_mapping_yml(mapping_path: Path) -> list[ObjectMapping]:
    """
    mapping.yml 을 파싱하여 ObjectMapping 목록을 반환한다.

    형식:
      {object_name}:
        gt:
          urdf_path: "..."
          z_rotation: 0
        {predictor}:
          urdf_path: "..."
          z_rotation: 0
          joint_pairs:
            {gt_joint_name}: {pred_joint_name}
            ...
    """
    import yaml

    base_dir = mapping_path.parent

    with mapping_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    objects: list[ObjectMapping] = []

    for obj_name, obj_data in raw.items():
        if not isinstance(obj_data, dict):
            continue

        # GT 파싱
        gt_raw = obj_data.get("gt")
        if gt_raw is None:
            print(f"[WARN] '{obj_name}': gt 블록이 없습니다. skip.")
            continue

        gt_path = (base_dir / gt_raw["urdf_path"]).resolve()
        if not gt_path.exists():
            print(f"[WARN] '{obj_name}': GT URDF 없음 → {gt_path}. skip.")
            continue

        gt_cfg = GTConfig(
            urdf_path=gt_path,
            z_rotation_deg=float(gt_raw.get("z_rotation", 0)),
        )

        # Predictor 파싱
        predictors: dict[str, PredictorConfig] = {}
        for key, val in obj_data.items():
            if key == "gt" or not isinstance(val, dict):
                continue

            pred_path_raw = val.get("urdf_path")
            if pred_path_raw is None:
                continue

            pred_path = (base_dir / pred_path_raw).resolve()
            if not pred_path.exists():
                print(f"[WARN] '{obj_name}/{key}': URDF 없음 → {pred_path}. skip.")
                continue

            joint_pairs: dict[str, str] = {}
            raw_pairs = val.get("joint_pairs", {})
            if raw_pairs:
                for gt_j, pred_j in raw_pairs.items():
                    joint_pairs[str(gt_j)] = str(pred_j)

            predictors[key] = PredictorConfig(
                urdf_path=pred_path,
                z_rotation_deg=float(val.get("z_rotation", 0)),
                joint_pairs=joint_pairs,
            )

        objects.append(ObjectMapping(name=obj_name, gt=gt_cfg, predictors=predictors))

    return objects
