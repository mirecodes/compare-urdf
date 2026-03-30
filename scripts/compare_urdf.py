from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict
import numpy as np


# ---------------------------------------------------------------------------
# 데이터 구조체
# ---------------------------------------------------------------------------

@dataclass
class JointInfo:
    name: str
    joint_type: str
    origin_xyz: tuple[float, float, float]   # (x, y, z) in metres
    origin_rpy: tuple[float, float, float]   # (r, p, y) in radians
    axis_xyz: tuple[float, float, float]     # unit (or raw) direction vector


@dataclass
class ComparisonResult:
    joint_name_gt: str
    joint_name_pred: str

    # --- joint type ---
    type_gt: str
    type_pred: str
    type_match: bool

    # --- joint origin (xyz) ---
    origin_gt: tuple[float, float, float]
    origin_pred: tuple[float, float, float]
    origin_dist_m: float      # Type-appropriate distance (Perp for Revolute, L2 for others)

    # --- joint axis  ---
    axis_gt: tuple[float, float, float]
    axis_pred: tuple[float, float, float]
    axis_dot: float           # |dot product|  ∈ [0, 1]
    axis_angle_deg: float     # angle between axes [deg]  ∈ [0, 90]

    notes: list[str] = field(default_factory=list)


import yourdfpy


MOVABLE_TYPES = {"revolute", "prismatic", "continuous", "planar", "floating"}
REVOLUTE_TYPES = {"revolute", "continuous"}


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def load_movable_joints(urdf_path: Path) -> list[JointInfo]:
    """
    yourdfpy 라이브러리를 사용하여 URDF 의 movable joint 목록을 파싱하고,
    Root 링크 기준(World frame) 좌표를 반환한다.
    """
    model = yourdfpy.URDF.load(str(urdf_path))
    
    movable_joints: list[JointInfo] = []
    
    # yourdfpy는 내부적으로 씬 그래프를 구축하여 world_T_link 를 쉽게 계산한다.
    for joint_name, joint in model.joint_map.items():
        # MOVABLE_TYPES = {"revolute", "prismatic", "continuous", "planar", "floating"}
        if joint.type not in MOVABLE_TYPES:
            continue
            
        # Joint frame의 World Transform 계산
        # yourdfpy에서 joint.origin은 parent -> joint_frame 트랜스폼임.
        # world_T_joint = world_T_parent * parent_T_joint
        parent_link = joint.parent
        world_T_parent = model.get_transform(parent_link)
        world_T_joint = world_T_parent @ joint.origin
        
        # World Origin (xyz)
        world_xyz = tuple(world_T_joint[:3, 3])
        
        # World Axis (xyz)
        # URDF axis는 joint frame 기준으로 정의됨.
        local_axis = np.array(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0])
        world_axis = world_T_joint[:3, :3] @ local_axis
        world_axis = _normalize(tuple(world_axis))
        
        movable_joints.append(JointInfo(
            name=joint_name,
            joint_type=joint.type,
            origin_xyz=world_xyz,
            origin_rpy=(0,0,0), # World frame 이므로 rpy 0 처리
            axis_xyz=world_axis
        ))
        
    # 조인트 타입 순서 등을 일정하게 유지하기 위해 이름순 정렬 (필요시)
    movable_joints.sort(key=lambda x: x.name)
    return movable_joints


# ---------------------------------------------------------------------------
# 비교 로직
# ---------------------------------------------------------------------------

def _l2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _axis_angle(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    """
    두 방향벡터 사이의 |내적| 과 각도(deg)를 반환한다.
    부호는 무시한다 (축 방향은 ±180° 동일).
    반환: (|dot|, angle_deg)  angle_deg ∈ [0, 90]
    """
    na = np.array(a)
    nb = np.array(b)
    dot = np.dot(na, nb)
    dot_abs: float = abs(dot) if abs(dot) <= 1.0 else 1.0   # 수치 오차 방지
    angle_deg = math.degrees(math.acos(dot_abs))
    return dot_abs, angle_deg


def _perp_dist(p_gt: tuple[float, float, float], p_pred: tuple[float, float, float], axis_gt: tuple[float, float, float]) -> float:
    """GT 원점과 축 방향 벡터가 정의하는 직선과 Pred 원점 사이의 최단 거리."""
    pg = np.array(p_gt)
    pp = np.array(p_pred)
    ag = np.array(axis_gt)
    
    # Perp dist = ||(pp - pg) x ag|| (ag는 단위 벡터여야 함)
    v = pp - pg
    perp_v = np.cross(v, ag)
    return float(np.linalg.norm(perp_v))


def _line_dist(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> float:
    """
    두 직선 L1: p1 + t*d1, L2: p2 + s*d2 사이의 최단 거리를 계산한다.
    d1, d2 는 단위 방향 벡터임을 가정함.
    """
    w = p1 - p2
    b = np.dot(d1, d2)
    dist_sq = 1.0 - b*b
    
    # 두 축이 평행한 경우 (d1 x d2 ≈ 0)
    if dist_sq < 1e-9:
        perp_v = w - np.dot(w, d1) * d1
        return float(np.linalg.norm(perp_v))
    
    # 일반적인 경우 (Skew lines)
    d = np.dot(d1, w)
    e = np.dot(d2, w)
    sc = (b*e - d) / dist_sq
    tc = (e - b*d) / dist_sq
    res_v = w + sc*d1 - tc*d2
    return float(np.linalg.norm(res_v))


def compare_single_joint(gt: JointInfo, pred: JointInfo) -> ComparisonResult:
    """GT joint 하나와 pred joint 하나를 비교한다."""
    # revolute 와 continuous 는 동일한 것으로 간주
    if gt.joint_type in REVOLUTE_TYPES and pred.joint_type in REVOLUTE_TYPES:
        type_match = True
    else:
        type_match = (gt.joint_type == pred.joint_type)

    dot_abs, angle_deg = _axis_angle(gt.axis_xyz, pred.axis_xyz)
    
    # 조인트 타입에 따른 거리 계산 방식 결정
    if gt.joint_type in REVOLUTE_TYPES:
        # 회전 조인트는 축(Line)과의 수직 거리를 측정
        origin_dist = _perp_dist(gt.origin_xyz, pred.origin_xyz, gt.axis_xyz)
    else:
        # 그 외(Prismatic 등)는 단순 점간 거리
        origin_dist = _l2(gt.origin_xyz, pred.origin_xyz)

    notes: list[str] = []
    if not type_match:
        notes.append(
            f"Type mismatch: gt={gt.joint_type!r}, pred={pred.joint_type!r}"
        )

    return ComparisonResult(
        joint_name_gt=gt.name,
        joint_name_pred=pred.name,
        type_gt=gt.joint_type,
        type_pred=pred.joint_type,
        type_match=type_match,
        origin_gt=gt.origin_xyz,
        origin_pred=pred.origin_xyz,
        origin_dist_m=origin_dist,
        axis_gt=gt.axis_xyz,
        axis_pred=pred.axis_xyz,
        axis_dot=dot_abs,
        axis_angle_deg=angle_deg,
        notes=notes,
    )


def compare_urdf_files(gt_path: Path, pred_path: Path) -> ComparisonResult:
    """
    GT URDF 와 pred URDF 를 비교한다.

    매칭 전략:
      - joint 이름이 달라도 OK.
      - GT 의 첫 번째 movable joint 를 기준으로, Pred 의 모든 movable joint 중
        축 사이의 '직선 최단 거리'가 가장 가까운 하나를 선택하여 비교한다.
    """
    gt_joints   = load_movable_joints(gt_path)
    pred_joints = load_movable_joints(pred_path)

    if not gt_joints:
        raise ValueError(f"GT URDF 에 movable joint 가 없습니다: {gt_path}")

    gt = gt_joints[0]

    if not pred_joints:
        # Pred에 조인트가 없는 경우: 에러 대신 'none' 타입을 반환하여 비교 수행
        return ComparisonResult(
            joint_name_gt=gt.name,
            joint_name_pred="none",
            type_gt=gt.joint_type,
            type_pred="none",
            type_match=False,
            origin_gt=gt.origin_xyz,
            origin_pred=(0.0, 0.0, 0.0),
            origin_dist_m=float('nan'),
            axis_gt=gt.axis_xyz,
            axis_pred=(0.0, 0.0, 0.0),
            axis_dot=float('nan'),
            axis_angle_deg=float('nan'),
            notes=["No movable joint found in pred URDF"],
        )

    # 매칭 전략: GT(첫 번째) 축과 가장 가까운 Pred 축을 찾음
    p_gt = np.array(gt.origin_xyz)
    d_gt = np.array(gt.axis_xyz)
    
    best_pred = pred_joints[0]
    min_dist = float('inf')
    
    for p in pred_joints:
        dist = _line_dist(p_gt, d_gt, np.array(p.origin_xyz), np.array(p.axis_xyz))
        if dist < min_dist:
            min_dist = dist
            best_pred = p
            
    pred = best_pred

    # 매칭된 조인트가 1개 이상이고 이름이 다를 경우 알림
    if gt.name != pred.name:
        print(
            f"[INFO] Joint matched by distance: "
            f"GT={gt.name!r} ({gt.joint_type}) ↔ pred={pred.name!r} ({pred.joint_type}) "
            f"dist={min_dist:.4f}m"
        )

    return compare_single_joint(gt, pred)


# ---------------------------------------------------------------------------
# 결과 포매팅
# ---------------------------------------------------------------------------

def result_to_summary_dict(r: ComparisonResult) -> dict:
    """YAML 리포트 상단 요약 — 핵심 수치만."""
    res: dict[str, Any] = {
        "type_match": bool(r.type_match),
    }
    
    if not math.isnan(r.origin_dist_m):
        val = round(r.origin_dist_m, 4)
        if r.type_gt == "prismatic":
            res["origin_dist_m"] = f"({val:.4f})"
        else:
            res["origin_dist_m"] = float(val)
        
    if not math.isnan(r.axis_angle_deg):
        res["axis_angle_deg"] = float(round(r.axis_angle_deg, 2))  # type: ignore[call-overload]
    return res


def result_to_detail_dict(r: ComparisonResult) -> dict:
    """YAML 리포트 하단 상세 정보."""
    return {
        "joint_name_gt": r.joint_name_gt,
        "joint_name_pred": r.joint_name_pred,
        "type_gt": r.type_gt,
        "type_pred": r.type_pred,
        "origin_gt_xyz": [float(v) for v in r.origin_gt],
        "origin_pred_xyz": [float(v) for v in r.origin_pred],
        "axis_gt_xyz": [float(v) for v in r.axis_gt],
        "axis_pred_xyz": [float(v) for v in r.axis_pred],
        "axis_dot_abs": float(round(r.axis_dot, 6)),  # type: ignore[call-overload]
        "notes": r.notes,
    }


def result_to_dict(r: ComparisonResult) -> dict:
    """하위 호환용 통합 dict."""
    s = result_to_summary_dict(r)
    d = result_to_detail_dict(r)
    return {**s, **d}


def print_result(label: str, r: ComparisonResult) -> None:
    PASS = "\033[92m✔\033[0m"
    FAIL = "\033[91m✘\033[0m"

    type_icon = PASS if r.type_match else FAIL
    angle_icon = PASS if r.axis_angle_deg <= 10.0 else FAIL   # 10° 이내 pass

    print(f"\n{'─'*55}")
    print(f"  Prediction: {label}")
    print(f"  GT joint  : {r.joint_name_gt}  |  Pred joint: {r.joint_name_pred}")
    print(f"{'─'*55}")
    print(f"  {type_icon} Joint Type   gt={r.type_gt!r:12s}  pred={r.type_pred!r}")

    if not math.isnan(r.origin_dist_m):
        dist_label = "Origin (Perp)" if r.type_gt in REVOLUTE_TYPES else "Origin (L2)"
        val_str = f"({r.origin_dist_m:.4f} m)" if r.type_gt == "prismatic" else f"{r.origin_dist_m:.4f} m"
        print(f"  {'  '} {dist_label:15s} {val_str}")
        print(f"     gt  : {r.origin_gt}")
        print(f"     pred: {r.origin_pred}")
    else:
        print(f"  {'  '} Origin Dist    N/A (No joint)")

    if not math.isnan(r.axis_angle_deg):
        print(f"  {angle_icon} Axis |dot|   {r.axis_dot:.4f}  →  angle = {r.axis_angle_deg:.2f}°")
        print(f"     gt  : {r.axis_gt}")
        print(f"     pred: {r.axis_pred}")
    else:
        print(f"  {'  '} Axis angle   N/A (No joint)")

    if r.notes:
        for note in r.notes:
            print(f"  [!] {note}")
    print(f"{'─'*55}")
