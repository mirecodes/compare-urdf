"""
compare_multi_joint.py
----------------------
multi-joint URDF 비교를 위한 핵심 로직.
- mapping.yml 기반 조인트 매핑
- Z-rotation (base) 보정
- Joint offsets (revolute/prismatic) 보정
- Missing joint 를 에러로 기록 (type_match=False)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yourdfpy
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))

from compare_urdf import (
    JointInfo,
    ComparisonResult,
    MOVABLE_TYPES,
    REVOLUTE_TYPES,
    compare_single_joint,
    _normalize,
)


@dataclass
class GTConfig:
    """mapping.yml 에서 파싱한 GT 설정."""
    urdf_path: Path
    z_rotation_deg: float
    joint_count: int = 0   # GT 내 전체 movable joint 개수


@dataclass
class PredictorConfig:
    """mapping.yml 에서 파싱한 Predictor 설정."""
    urdf_path: Path
    z_rotation_deg: float
    joint_pairs: dict[str, str]       # {gt_joint_name: pred_joint_name}
    joint_offsets: dict[str, float]   # {pred_joint_name: offset_value}


@dataclass
class ObjectMapping:
    """하나의 Object 에 대한 전체 매핑 설정."""
    name: str
    gt: GTConfig
    predictors: dict[str, PredictorConfig]  # {predictor_name: PredictorConfig}


def make_z_rotation_matrix(angle_deg: float) -> np.ndarray:
    """Z축 기준 회전 행렬(4x4) 생성."""
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])


def load_joints_filtered(
    urdf_path: Path,
    z_rotation_deg: float,
    joint_offsets: dict[str, float] | None = None,
) -> dict[str, JointInfo]:
    """
    URDF 를 로드하고, z_rotation 및 joint_offsets 를 적용한 후
    World-frame 기준의 JointInfo 딕셔너리를 반환한다.
    """
    if not urdf_path.exists():
        return {}

    # 1. yourdfpy 로드
    try:
        model = yourdfpy.URDF.load(str(urdf_path))
    except Exception as e:
        print(f"  [ERROR] URDF 로드 실패: {urdf_path.name} ({e})")
        return {}

    # 2. Joint Offsets (State) 적용
    # 이 오프셋은 yourdfpy 의 update_cfg 에 전달되어 자식 링크들의 transform 에 영향을 줌.
    if joint_offsets:
        cfg_to_update = {}
        for jname, val in joint_offsets.items():
            if jname not in model.joint_map:
                continue
            joint = model.joint_map[jname]
            if joint.type in REVOLUTE_TYPES:
                cfg_to_update[jname] = math.radians(val)
            else:
                cfg_to_update[jname] = val
        
        if cfg_to_update:
            model.update_cfg(configuration=cfg_to_update)

    # 3. 좌표계 변환 및 JointInfo 생성
    R_z = make_z_rotation_matrix(z_rotation_deg)
    all_joints: dict[str, JointInfo] = {}

    for jname, joint in model.joint_map.items():
        if joint.type not in MOVABLE_TYPES:
            continue

        # parent -> world 트랜스폼 가져오기
        parent_link = joint.parent
        world_T_parent = model.get_transform(parent_link)
        
        # Base z_rotation 적용 (Global 오브젝트 회전)
        # joint.origin 은 parent -> joint_frame (fixed)
        world_T_joint_base = R_z @ world_T_parent @ joint.origin
        
        world_xyz = list(world_T_joint_base[:3, 3])
        local_axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0])
        
        # 월드 좌표계에서의 축 벡터
        world_axis_vec = world_T_joint_base[:3, :3] @ local_axis
        world_axis_vec = _normalize(tuple(world_axis_vec))
        
        # Prismatic 인 경우, 오프셋에 의해 원점이 축 방향으로 이동함
        if joint_offsets and jname in joint_offsets and joint.type == "prismatic":
            off_val = joint_offsets[jname]
            world_xyz[0] += world_axis_vec[0] * off_val
            world_xyz[1] += world_axis_vec[1] * off_val
            world_xyz[2] += world_axis_vec[2] * off_val

        all_joints[jname] = JointInfo(
            name=jname,
            joint_type=joint.type,
            origin_xyz=tuple(world_xyz),
            origin_rpy=(0, 0, 0),
            axis_xyz=world_axis_vec
        )

    return all_joints


def _missing_joint_result(gt_name: str, pred_name: str, reason: str) -> ComparisonResult:
    """조인트를 찾을 수 없는 경우(missing)에 대한 에러 결과 반환."""
    return ComparisonResult(
        joint_name_gt=gt_name,
        joint_name_pred=pred_name,
        type_gt="unknown",
        type_pred="missing",
        type_match=False,
        origin_gt=(0.0, 0.0, 0.0),
        origin_pred=(0.0, 0.0, 0.0),
        origin_dist_m=float("nan"),
        axis_gt=(0.0, 0.0, 0.0),
        axis_pred=(0.0, 0.0, 0.0),
        axis_dot=float("nan"),
        axis_angle_deg=float("nan"),
        notes=[f"Joint missing: {reason}"],
    )


def compare_by_mapping(
    obj_mapping: ObjectMapping,
) -> dict[str, list[ComparisonResult]]:
    """ObjectMapping 설정을 바탕으로 GT 와 모든 Predictor 를 비교한다."""
    gt_cfg = obj_mapping.gt
    
    # GT 조인트 로드
    gt_joints = load_joints_filtered(gt_cfg.urdf_path, gt_cfg.z_rotation_deg)
    
    results: dict[str, list[ComparisonResult]] = {}
    
    for p_name, p_cfg in obj_mapping.predictors.items():
        # Predictor 조인트 로드 (오프셋 적용)
        pred_joints = load_joints_filtered(
            p_cfg.urdf_path, 
            p_cfg.z_rotation_deg, 
            p_cfg.joint_offsets
        )
        
        pair_results: list[ComparisonResult] = []
        for gt_name, pr_name in p_cfg.joint_pairs.items():
            if gt_name not in gt_joints:
                print(f"  [ERROR] GT 조인트 '{gt_name}' 가 파일에 없습니다.")
                pair_results.append(_missing_joint_result(gt_name, pr_name, f"GT joint '{gt_name}' not found"))
                continue
            
            if pr_name not in pred_joints:
                print(f"  [ERROR] Pred 조인트 '{pr_name}' 가 파일에 없습니다.")
                pair_results.append(_missing_joint_result(gt_name, pr_name, f"Pred joint '{pr_name}' not found"))
                continue
                
            # 실제 비교 수행
            r = compare_single_joint(gt_joints[gt_name], pred_joints[pr_name])
            pair_results.append(r)
            
        results[p_name] = pair_results
        
    return results


def parse_mapping_yml(mapping_path: Path) -> list[ObjectMapping]:
    """mapping.yml 을 파싱하여 ObjectMapping 리스트를 반환한다."""
    if not mapping_path.exists():
        return []
        
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
        
    if not raw:
        return []
        
    base_dir = mapping_path.parent
    mappings: list[ObjectMapping] = []
    
    for obj_name, conf in raw.items():
        if "gt" not in conf:
            continue
            
        # 1. GT 설정
        gt_raw = conf["gt"]
        gt_path = (base_dir / gt_raw["urdf_path"]).resolve()
        
        # GT movable joint 개수 미리 파악
        count = 0
        try:
            m = yourdfpy.URDF.load(str(gt_path))
            count = sum(1 for j in m.joint_map.values() if j.type in MOVABLE_TYPES)
        except:
            pass
            
        gt_cfg = GTConfig(
            urdf_path=gt_path,
            z_rotation_deg=float(gt_raw.get("z_rotation", 0)),
            joint_count=count
        )
        
        # 2. Predictors 설정
        predictors: dict[str, PredictorConfig] = {}
        for p_name, p_raw in conf.items():
            if p_name == "gt":
                continue
            
            if "urdf_path" not in p_raw:
                continue
                
            p_path = (base_dir / p_raw["urdf_path"]).resolve()
            if not p_path.exists():
                print(f"  [WARN] '{obj_name}/{p_name}': URDF 없음 skip.")
                continue
                
            jp = p_raw.get("joint_pairs", {})
            if not jp:
                jp = {}
                
            # 정규화: {gt: pred} 또는 {gt: {name: pred, offset: v}}
            pairs: dict[str, str] = {}
            offsets: dict[str, float] = {}
            
            for g_j, data in jp.items():
                if isinstance(data, dict):
                    p_j = data.get("name")
                    off = float(data.get("offset", 0))
                    pairs[g_j] = p_j
                    offsets[p_j] = off
                else:
                    pairs[g_j] = data
                    
            predictors[p_name] = PredictorConfig(
                urdf_path=p_path,
                z_rotation_deg=float(p_raw.get("z_rotation", 0)),
                joint_pairs=pairs,
                joint_offsets=offsets
            )
            
        if predictors:
            mappings.append(ObjectMapping(name=obj_name, gt=gt_cfg, predictors=predictors))
            
    return mappings
