import sys
import numpy as np
from pathlib import Path

# Add scripts directory to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))

from compare_urdf import JointInfo, compare_single_joint, _line_dist

def test_type_matching():
    print("Running Joint Type Matching Tests...")
    origin = (0.0, 0.0, 0.0)
    axis = (0.0, 0.0, 1.0)
    cases = [
        ("revolute", "revolute", True),
        ("continuous", "continuous", True),
        ("revolute", "continuous", True),
        ("continuous", "revolute", True),
        ("prismatic", "prismatic", True),
        ("revolute", "prismatic", False),
    ]
    passed = 0
    for gt_type, pred_type, expected in cases:
        gt = JointInfo(name="gt", joint_type=gt_type, origin_xyz=origin, origin_rpy=(0,0,0), axis_xyz=axis)
        pred = JointInfo(name="pred", joint_type=pred_type, origin_xyz=origin, origin_rpy=(0,0,0), axis_xyz=axis)
        result = compare_single_joint(gt, pred)
        if result.type_match == expected:
            passed += 1
    print(f"  Type matching: {passed}/{len(cases)} passed.")
    return passed == len(cases)

def test_line_dist():
    print("Running Line Distance Tests...")
    # Case 1: Parallel lines
    p1 = np.array([0, 0, 0])
    d1 = np.array([0, 0, 1])
    p2 = np.array([1, 0, 0])
    d2 = np.array([0, 0, 1])
    dist = _line_dist(p1, d1, p2, d2)
    c1 = abs(dist - 1.0) < 1e-6
    print(f"  Parallel: {dist:.4f} (Expected: 1.0) {'✔' if c1 else '✘'}")

    # Case 2: Skew lines (X-axis and Y-axis separated by Z=1)
    p1 = np.array([0, 0, 0])
    d1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 1])
    d2 = np.array([0, 1, 0])
    dist = _line_dist(p1, d1, p2, d2)
    c2 = abs(dist - 1.0) < 1e-6
    print(f"  Skew:     {dist:.4f} (Expected: 1.0) {'✔' if c2 else '✘'}")

    # Case 3: Intersecting lines
    p1 = np.array([0, 0, 0])
    d1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    d2 = np.array([0, 1, 0])
    dist = _line_dist(p1, d1, p2, d2)
    c3 = abs(dist - 0.0) < 1e-6
    print(f"  Intersect:{dist:.4f} (Expected: 0.0) {'✔' if c3 else '✘'}")

    return c1 and c2 and c3

if __name__ == "__main__":
    t1 = test_type_matching()
    t2 = test_line_dist()
    if t1 and t2:
        print("\nALL TESTS PASSED!")
    else:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)
