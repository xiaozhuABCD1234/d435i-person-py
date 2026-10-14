# SPDX-License-Identifier: AGPL-3.0-or-later
"""config.toml 读取与命令行覆盖的单元测试 —— 不碰相机、不加载模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from d435i_person.config import (
    CLI_FIELDS,
    CONFIG_FIELDS,
    Settings,
    apply_cli,
    filter_spec,
    load_settings,
    resolve_path,
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    settings, warnings = load_settings(tmp_path / "nope.toml")
    assert settings.detect_hz == Settings().detect_hz
    assert warnings and "没找到配置文件" in warnings[0]


def test_values_are_read_and_coerced(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
[camera]
serial = "117322075393"
color_size = "848x480"
fps = 60
exposure_mode = "manual"

[filters]
spatial = true

[world]
eye_typo = 1
height = 1.25
pitch_deg = -15
pose3d_eye = [1, 2, 3]
""",
    )
    settings, warnings = load_settings(path)
    assert settings.serial == "117322075393"
    assert settings.color_size == "848x480"
    assert settings.fps == 60
    assert settings.exposure_mode == "manual"
    assert settings.spatial is True
    assert settings.temporal is False
    assert settings.world_height == 1.25
    assert settings.world_pitch_deg == -15.0
    # 未知键（[world].eye_typo）必须警告，不能静默忽略
    assert any("eye_typo" in w for w in warnings)
    # [world].pose3d_eye 在 [world] 段里也不合法 -> 也要警告
    assert any("pose3d_eye" in w for w in warnings)


def test_bad_type_warns_and_keeps_default(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
[camera]
fps = "很快"

[detector]
conf = true
""",
    )
    settings, warnings = load_settings(path)
    assert settings.fps == Settings().fps  # 类型不对 -> 用默认值
    assert settings.conf == Settings().conf
    assert len([w for w in warnings if "不合法" in w]) == 2


def test_three_number_tuple_required(tmp_path: Path) -> None:
    path = write(tmp_path, "[pose3d]\neye = [0.0, 0.0]\n")
    settings, warnings = load_settings(path)
    assert settings.pose3d_eye == Settings().pose3d_eye
    assert any("3 个数字" in w for w in warnings)


def test_cli_overrides_config(tmp_path: Path) -> None:
    path = write(tmp_path, "[run]\ndetect_hz = 5.0\n")
    settings, _ = load_settings(path)
    args = argparse.Namespace(detect_hz=30.0, world_enabled=None, conf=None)
    settings = apply_cli(settings, args)
    assert settings.detect_hz == 30.0
    # 命令行没给的（None）不能把配置里的值冲掉
    assert settings.world_enabled is False


def test_every_config_field_maps_to_a_settings_attribute() -> None:
    settings = Settings()
    for (section, key), attr in CONFIG_FIELDS.items():
        assert hasattr(settings, attr), f"[{section}].{key} -> 不存在的字段 {attr}"
        assert not attr.startswith("_")


def test_cli_fields_come_from_config_fields() -> None:
    assert set(CLI_FIELDS) == set(CONFIG_FIELDS.values())


def test_filter_spec_shape() -> None:
    settings = Settings(spatial=True, hole_filling=True)
    assert filter_spec(settings) == {
        "spatial": True,
        "temporal": False,
        "hole_filling": True,
    }


def test_resolve_path_is_relative_to_project_root() -> None:
    resolved = resolve_path("models/yolo26n-pose.pt")
    assert resolved.is_absolute()
    assert resolved.name == "yolo26n-pose.pt"
    assert (resolved.parent.parent / "pyproject.toml").is_file()


def test_default_weights_exist() -> None:
    assert resolve_path(Settings().weights).is_file()


@pytest.mark.parametrize("bad", [1, "x", 3.5])
def test_non_table_section_warns(tmp_path: Path, bad) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"camera = {bad!r}\n", encoding="utf-8")
    _, warnings = load_settings(path)
    assert any("不是表" in w for w in warnings)
