# SPDX-License-Identifier: AGPL-3.0-or-later
"""跟踪（卡尔曼 + IoU 关联 + id/轨迹）的单元测试 —— 不需要相机、不加载模型。"""

from __future__ import annotations

import numpy as np
import pytest

from d435i_person.geometry import Intrinsics
from d435i_person.pose import NUM_KEYPOINTS, PoseInstance
from d435i_person.track import PersonTracker, ScalarKalman, iou

K = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
CONF = np.full(NUM_KEYPOINTS, 0.9, dtype=np.float32)


def make_pose(x: float = 320.0, y: float = 240.0, conf: float = 0.8) -> PoseInstance:
    """一个 40×300 的框，关键点集中在框中间一条竖线上。"""
    points = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
    points[:, 0] = x - 4 + np.arange(NUM_KEYPOINTS) * 0.5
    points[:, 1] = y + np.arange(NUM_KEYPOINTS)
    return PoseInstance(
        conf=conf,
        xyxy=(x - 20.0, y - 150.0, x + 20.0, y + 150.0),
        keypoints=points,
        kpt_conf=CONF.copy(),
    )


def flat_depth(value: float = 2.0) -> np.ndarray:
    return np.full((480, 640), value, dtype=np.float32)


def tracker(**kwargs) -> PersonTracker:
    defaults = {
        "kalman": True,
        "iou_threshold": 0.3,
        "max_missed": 5,
        "max_missed_kpt": 3,
        "trail_length": 100,
        "trail_interval": 0.1,
    }
    defaults.update(kwargs)
    return PersonTracker(**defaults)


# ------------------------------------------------------------------------ IoU
def test_iou_identical_is_one_and_disjoint_is_zero() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# --------------------------------------------------------------------- Kalman
def test_kalman_holds_a_constant_signal() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.01)
    kf.init(5.0)
    for _ in range(10):
        kf.predict(0.1)
        kf.update(5.0)
    assert kf.position == pytest.approx(5.0, abs=1e-3)


def test_kalman_follows_a_ramp_with_lag() -> None:
    kf = ScalarKalman(process_noise=1.0, measurement_noise=0.05)
    kf.init(0.0)
    for step in range(1, 30):
        kf.predict(1.0)
        kf.update(float(step))
    assert 25.0 < kf.position < 29.5


def test_kalman_smooths_a_noisy_measurement() -> None:
    rng = np.random.default_rng(0)
    kf = ScalarKalman(process_noise=0.1, measurement_noise=0.5)
    kf.init(0.0)
    filtered: list[float] = []
    raw: list[float] = []
    for _ in range(60):
        kf.predict(0.05)
        noisy = 1.0 + float(rng.normal(scale=0.3))
        kf.update(noisy)
        filtered.append(kf.position)
        raw.append(noisy)
    # 滤波后的波动必须比原始观测小（同样的样本，只比后半段，躲过初始化瞬态）
    filtered_error = np.abs(np.array(filtered[10:]) - 1.0).mean()
    raw_error = np.abs(np.array(raw[10:]) - 1.0).mean()
    assert filtered_error < raw_error


# ---------------------------------------------------------------- id 与轨迹
def test_same_person_keeps_its_id() -> None:
    track = tracker()
    for step in range(5):
        people = track.update(
            [make_pose(320.0 + step)], flat_depth(2.0), K, threshold=0.5, t=step * 0.1
        )
        assert [person.id for person in people] == [1]
    person = people[0]
    assert person.age == 5
    assert person.missed == 0
    assert person.valid_count == NUM_KEYPOINTS
    assert person.distance == pytest.approx(np.linalg.norm(person.position))


def test_two_people_get_different_ids() -> None:
    track = tracker()
    people = track.update(
        [make_pose(200.0), make_pose(450.0)], flat_depth(2.0), K, threshold=0.5, t=0.0
    )
    assert [person.id for person in people] == [1, 2]


def test_missing_person_is_predicted_then_dropped_and_id_is_not_reused() -> None:
    track = tracker(max_missed=2)
    people = track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.0)
    # 第一帧就是命中帧，不该被当成漏检
    assert (people[0].missed, people[0].age) == (0, 1)

    # 漏检 1 帧：还在，标成 missed，位置靠卡尔曼预测，2D 关键点不再画
    people = track.update([], flat_depth(2.0), K, threshold=0.5, t=0.1)
    assert len(people) == 1
    assert people[0].missed == 1
    assert np.isfinite(people[0].position).all()
    assert people[0].kpt_conf.max() == 0.0

    # 漏检到 max_missed 帧还能留住，再多一帧就删
    assert len(track.update([], flat_depth(2.0), K, threshold=0.5, t=0.2)) == 1
    assert track.update([], flat_depth(2.0), K, threshold=0.5, t=0.3) == []

    # 新人拿到新 id（旧 id 永不复用）
    people = track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.4)
    assert [person.id for person in people] == [2]


def test_trail_grows_with_time_and_respects_interval() -> None:
    track = tracker(trail_interval=0.1)
    track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.0)
    assert len(track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.05)[0].trail) == 1
    assert len(track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.1)[0].trail) == 2


def test_trail_can_be_disabled_and_is_capped() -> None:
    people = tracker(trail_length=0).update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.0)
    assert len(people[0].trail) == 0

    track = tracker(trail_length=3, trail_interval=0.0)
    for step in range(6):
        people = track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=step * 0.1)
    assert len(people[0].trail) == 3  # deque 只留最近 3 个点


def test_walking_person_trail_is_monotonic_in_x() -> None:
    track = tracker(trail_interval=0.0)
    for step in range(5):
        people = track.update(
            [make_pose(300.0 + step * 10)], flat_depth(2.0), K, threshold=0.5, t=step * 0.1
        )
    xs = people[0].trail[:, 0]
    assert np.all(np.diff(xs) > 0)


# --------------------------------------------------------- 3D 距离门限与开关
def test_match_max_distance_prevents_swallowing_a_far_person() -> None:
    # 框完全重合（IoU=1），但深度从 2 m 跳到 8 m：默认会被当成同一人
    loose = tracker()
    loose.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.0)
    people = loose.update([make_pose()], flat_depth(8.0), K, threshold=0.5, t=0.1)
    assert [person.id for person in people] == [1]

    strict = tracker(match_max_distance=1.0)
    strict.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.0)
    people = strict.update([make_pose()], flat_depth(8.0), K, threshold=0.5, t=0.1)
    assert sorted(person.id for person in people) == [1, 2]


def test_without_kalman_joints_come_straight_from_the_observation() -> None:
    track = tracker(kalman=False)
    people = track.update([make_pose()], flat_depth(3.0), K, threshold=0.5, t=0.0)
    assert np.allclose(people[0].joints[:, 2], 3.0)
    assert people[0].valid_count == NUM_KEYPOINTS


def test_kalman_keeps_joints_and_position_on_the_same_depth() -> None:
    track = tracker()
    people = track.update([make_pose()], flat_depth(4.0), K, threshold=0.5, t=0.0)
    # 第一帧的滤波值就是观测值
    assert np.allclose(people[0].joints[:, 2], 4.0, atol=1e-3)
    assert people[0].position[2] == pytest.approx(4.0, abs=1e-3)


def test_no_depth_means_no_position_but_still_tracked() -> None:
    track = tracker()
    people = track.update(
        [make_pose()], np.full((480, 640), np.nan, np.float32), K, threshold=0.5, t=0.0
    )
    assert people[0].valid_count == 0
    # 位置未被观测初始化过 -> nan（绝不能报成原点 (0,0,0)）
    assert np.isnan(people[0].position).all()
    assert np.isnan(people[0].distance)
    assert len(people[0].trail) == 0


def test_position_filter_initializes_on_the_first_valid_frame() -> None:
    track = tracker()
    empty = np.full((480, 640), np.nan, np.float32)
    track.update([make_pose()], empty, K, threshold=0.5, t=0.0)
    people = track.update([make_pose()], flat_depth(2.0), K, threshold=0.5, t=0.1)
    assert people[0].position[2] == pytest.approx(2.0, abs=1e-3)
    assert len(people[0].trail) == 1
