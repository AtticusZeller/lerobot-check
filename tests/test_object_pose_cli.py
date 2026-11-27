from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_object_pose_edges import (
    IMAGE_SIZE,
    _bbox_with_margin,
    _project_rect,
    _RasterSegmenter,
    _rect_xy,
    _single_arm,
    _two_arm_calibration,
)

from lerobot_check.cli import main
from lerobot_check.object_pose import (
    EpisodeObjectMeasurement,
    ObjectPoseMeasurement,
    apply_size_outlier_filter,
    extract_dataset_object_poses,
    object_frame_tool_offsets,
    parse_prompt_box,
    parse_prompt_boxes,
)
from lerobot_check.schema import CoverageSpec, ObjectSpec, SubtaskSpec, TaskSpec, save_task_spec

PLANE_Z = -0.40
TRUTHS = {0: (0.80, 0.05, 20.0), 1: (0.81, 0.06, 50.0), 2: (0.79, 0.04, -10.0)}
MOVED = (0.70, -0.10, 30.0)


def _polygons(calib) -> dict[int, np.ndarray]:
    polys = {
        ep: _project_rect(calib, "right", _rect_xy(x, y, 0.14, 0.07, yaw), PLANE_Z)
        for ep, (x, y, yaw) in TRUTHS.items()
    }
    polys[3] = _project_rect(calib, "right", _rect_xy(*MOVED[:2], 0.14, 0.07, MOVED[2]), PLANE_Z)
    return polys


def _union_box(polys: list[np.ndarray]) -> tuple[int, int, int, int]:
    boxes = [_bbox_with_margin(p, 20) for p in polys]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _spec(layout_box: tuple[int, int, int, int]) -> TaskSpec:
    return TaskSpec(
        name="demo",
        subtasks=(SubtaskSpec(name="pick", prompt="Pick."),),
        objects=(
            ObjectSpec(
                "phone",
                "pick",
                "grasp",
                "right",
                360.0,
                layout_box=layout_box,
                object_plane_z_m=PLANE_Z,
            ),
            ObjectSpec("loose", "pick", "grasp", "right", 180.0),
        ),
        coverage=CoverageSpec(xy_bin_m=0.05, yaw_bin_deg=30.0, sparse_threshold=2),
    )


def _setup(monkeypatch: pytest.MonkeyPatch):
    calib = _single_arm(_two_arm_calibration(), "right")
    polys = _polygons(calib)
    layout_box = _union_box([polys[ep] for ep in TRUTHS])
    override = _bbox_with_margin(polys[3], 15)
    by_episode = {ep: {layout_box: polys[ep]} for ep in TRUTHS}
    by_episode[3] = {override: polys[3]}
    images = {ep: Image.new("RGB", IMAGE_SIZE, color=(ep, 0, 0)) for ep in range(4)}
    monkeypatch.setattr(
        "lerobot_check.object_pose.decode_frame_zero_images",
        lambda _ds, _cam, eps: {ep: images[ep] for ep in eps},
    )
    return calib, polys, layout_box, override, by_episode


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("33:box_lid=1,2,30,40", (33, "box_lid", (1, 2, 30, 40))),
        ("0:a b=0,0,1,1", (0, "a b", (0, 0, 1, 1))),
    ],
)
def test_parse_prompt_box(text: str, expected: tuple) -> None:
    assert parse_prompt_box(text) == expected


@pytest.mark.parametrize(
    "text",
    ["33box=1,2,3,4", "33:box", "x:box=1,2,3,4", "3:box=1,2,3", "3:=1,2,3,4", "3:box=1,2,3,4.5"],
)
def test_parse_prompt_box_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="EP:OBJECT"):
        parse_prompt_box(text)


def test_parse_prompt_boxes_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_prompt_boxes(["1:a=0,0,5,5", "1:a=1,1,6,6"])


def test_prompt_box_replaces_layout_box_for_one_episode_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calib, _, layout_box, override, by_episode = _setup(monkeypatch)
    segmenter = _RasterSegmenter(by_episode)
    payload, _ = extract_dataset_object_poses(
        SimpleNamespace(repo_id="Org/demo", revision="rev1"),  # ty: ignore[invalid-argument-type]
        _spec(layout_box),
        calib,
        calibration_sha256="a" * 64,
        episodes=[0, 1, 2, 3],
        plane_z_by_object={"phone": PLANE_Z},
        segmenter=segmenter,  # ty: ignore[invalid-argument-type]
        prompt_boxes={(3, "phone"): override},
    )
    episodes = {e["episode_index"]: e for e in payload["objects"]["phone"]["episodes"]}
    assert payload["objects"]["phone"]["layout_box"] == list(layout_box)
    assert [("prompt_box" in episodes[ep]) for ep in range(4)] == [False, False, False, True]
    assert episodes[3]["prompt_box"] == list(override)
    assert episodes[3]["status"] == "ok"
    assert math.hypot(episodes[3]["x_m"] - MOVED[0], episodes[3]["y_m"] - MOVED[1]) < 0.004


def test_prompt_box_without_override_skips_moved_object(monkeypatch: pytest.MonkeyPatch) -> None:
    calib, _, layout_box, _, by_episode = _setup(monkeypatch)
    payload, _ = extract_dataset_object_poses(
        SimpleNamespace(repo_id="Org/demo", revision="rev1"),  # ty: ignore[invalid-argument-type]
        _spec(layout_box),
        calib,
        calibration_sha256="a" * 64,
        episodes=[0, 1, 2, 3],
        plane_z_by_object={"phone": PLANE_Z},
        segmenter=_RasterSegmenter(by_episode),  # ty: ignore[invalid-argument-type]
    )
    assert payload["objects"]["phone"]["episodes"][3]["status"] == "empty mask"


def test_size_filter_keeps_prompt_box_on_skipped_record() -> None:
    def rec(ep: int, length: float, box=None) -> EpisodeObjectMeasurement:
        pose = ObjectPoseMeasurement(0.5, 0.1, -0.4, 10.0, 10.0, length, length / 2)
        return EpisodeObjectMeasurement(ep, 0, "ok", [[0, 0], [1, 0], [1, 1]], pose, box)

    filtered = apply_size_outlier_filter([rec(0, 0.1), rec(1, 0.1), rec(2, 0.3, (1, 2, 3, 4))])
    assert filtered[2].status.startswith("rectangle size")
    assert filtered[2].prompt_box == (1, 2, 3, 4)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({(0, "nope"): (0, 0, 10, 10)}, "unknown object"),
        ({(0, "loose"): (0, 0, 10, 10)}, "unknown object"),
        ({(9, "phone"): (0, 0, 10, 10)}, "not among the processed episodes"),
        ({(0, "phone"): (600, 0, 641, 10)}, "must satisfy"),
        ({(0, "phone"): (0, 10, 10, 10)}, "must satisfy"),
        ({(0, "phone"): (-1, 0, 10, 10)}, "must satisfy"),
    ],
)
def test_prompt_box_validation(monkeypatch: pytest.MonkeyPatch, override: dict, match: str) -> None:
    calib, _, layout_box, _, by_episode = _setup(monkeypatch)
    with pytest.raises(ValueError, match=match):
        extract_dataset_object_poses(
            SimpleNamespace(repo_id="Org/demo", revision="rev1"),  # ty: ignore[invalid-argument-type]
            _spec(layout_box),
            calib,
            calibration_sha256="a" * 64,
            episodes=[0, 1, 2],
            plane_z_by_object={"phone": PLANE_Z},
            segmenter=_RasterSegmenter(by_episode),  # ty: ignore[invalid-argument-type]
            prompt_boxes=override,
        )


def _write_calib(calib, path: Path) -> None:
    arm = calib.arms["right"]
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "repo_id": "Org/demo",
                "revision": "rev1",
                "camera": calib.camera,
                "image_size": list(calib.image_size),
                "intrinsics": calib.intrinsics.raw,
                "arms": {
                    "right": {
                        "T_cam_base": arm.T_cam_base.tolist(),
                        "tool_offset": arm.tool_offset.tolist(),
                        "n_points": 10,
                        "rms_px": 1.0,
                        "loo_rms_px": 1.2,
                        "residuals": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calib, _, layout_box, override, by_episode = _setup(monkeypatch)
    ds_dir = tmp_path / "lerobot" / "Org" / "demo"
    ds_dir.mkdir(parents=True)
    spec_path = tmp_path / "spec.yaml"
    save_task_spec(_spec(layout_box), spec_path)
    calib_path = tmp_path / "head.json"
    _write_calib(calib, calib_path)
    fake_ds = SimpleNamespace(root=ds_dir.resolve(), repo_id="Org/demo", revision="rev1")
    fake_ds.total_episodes = 4
    monkeypatch.setattr("lerobot_check.object_pose.DatasetView", lambda *a, **kw: fake_ds)
    monkeypatch.setattr(
        "lerobot_check.object_pose.SamSegmenter", lambda device: _RasterSegmenter(by_episode)
    )
    base_args = [
        "object-poses",
        "--dataset-root",
        str(ds_dir),
        "--task-spec",
        str(spec_path),
        "--calib",
        str(calib_path),
        "-o",
        str(tmp_path / "poses.json"),
    ]
    return SimpleNamespace(tmp=tmp_path, args=base_args, override=override)


def _distribution_file(tmp: Path, tool_local_xy: tuple[float, float]) -> Path:
    poses = []
    for ep, (x, y, yaw) in {**TRUTHS, 3: MOVED}.items():
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        lx, ly = tool_local_xy
        poses.append(
            {
                "episode_index": ep,
                "arm": "right",
                "x_m": x + c * lx - s * ly,
                "y_m": y + s * lx + c * ly,
            }
        )
    path = tmp / "distribution.json"
    path.write_text(
        json.dumps(
            {
                "provenance": {
                    "repo_id": "Org/demo",
                    "revision": "rev1",
                    "position_point": "tool_point",
                },
                "objects": [{"name": "phone", "median_z_m": PLANE_Z, "poses": poses}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_object_poses_with_prompt_box_and_offsets(
    cli_env, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = _distribution_file(cli_env.tmp, (0.02, -0.01))
    box = ",".join(str(v) for v in cli_env.override)
    rc = main([*cli_env.args, "--prompt-box", f"3:phone={box}", "--distribution", str(dist)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "phone: 4/4 ok" in out
    sidecar = json.loads((cli_env.tmp / "poses.json").read_text(encoding="utf-8"))
    assert sidecar["objects"]["phone"]["episodes"][3]["prompt_box"] == list(cli_env.override)
    offsets = next(line for line in out.splitlines() if "tool offset in object frame" in line)
    assert "n=4" in offsets
    match = re.search(r"mean=\(([-+.\d]+), ([-+.\d]+)\) cm std=\(([.\d]+), ([.\d]+)\)", offsets)
    assert match is not None
    mean_x, mean_y, std_x, std_y = (float(v) for v in match.groups())
    assert (mean_x, mean_y) == pytest.approx((2.0, -1.0), abs=0.1)
    assert max(std_x, std_y) < 0.1


@pytest.mark.parametrize(
    ("extra", "tag", "message"),
    [
        (["--prompt-box", "3phone=1,2,3,4"], "[prompt-box]", "EP:OBJECT"),
        (["--prompt-box", "3:ghost=1,2,30,40"], "[object-poses]", "unknown object"),
        (["--prompt-box", "3:phone=1,2,700,40"], "[object-poses]", "must satisfy"),
        (["--episodes", "0-1", "--prompt-box", "3:phone=1,2,30,40"], "[object-poses]", "not among"),
        (["--episodes", "7"], "[episodes]", "outside"),
    ],
)
def test_cli_object_poses_error_paths(
    cli_env, capsys: pytest.CaptureFixture[str], extra: list[str], tag: str, message: str
) -> None:
    rc = main([*cli_env.args, *extra])
    err = capsys.readouterr().err
    assert rc == 1
    assert tag in err and message in err
    assert not (cli_env.tmp / "poses.json").exists()


def test_cli_object_poses_rejects_segmenter_failure(
    cli_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(device: str):
        raise RuntimeError(f"no {device} device")

    monkeypatch.setattr("lerobot_check.object_pose.SamSegmenter", broken)
    rc = main([*cli_env.args, "--device", "cuda"])
    assert rc == 1
    assert "[object-poses] no cuda device" in capsys.readouterr().err


def test_cli_object_poses_rejects_distribution_for_other_dataset(
    cli_env, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = _distribution_file(cli_env.tmp, (0.0, 0.0))
    payload = json.loads(dist.read_text(encoding="utf-8"))
    payload["provenance"]["revision"] = "other"
    dist.write_text(json.dumps(payload), encoding="utf-8")
    rc = main([*cli_env.args, "--distribution", str(dist)])
    assert rc == 1
    assert "does not match dataset" in capsys.readouterr().err


def _object_and_tool(ep: int, x: float, y: float, yaw: float, local: tuple[float, float]):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    obj = {"episode_index": ep, "arm": "right", "x_m": x, "y_m": y, "raw_yaw_deg": yaw % 180.0}
    tool = {
        "episode_index": ep,
        "arm": "right",
        "x_m": x + c * local[0] - s * local[1],
        "y_m": y + s * local[0] + c * local[1],
    }
    return obj, tool


def test_object_frame_offsets_are_constant_across_rotation_and_wraparound() -> None:
    local = (0.03, -0.012)
    yaws = [-8.0, -2.0, 4.0, 25.0, 52.0]
    pairs = [
        _object_and_tool(ep, 0.8 + 0.01 * ep, 0.1 - 0.02 * ep, yaw, local)
        for ep, yaw in enumerate(yaws)
    ]
    objects = [obj for obj, _ in pairs]
    stray = {"episode_index": 99, "arm": "right", "x_m": 0, "y_m": 0}
    tools = [tool for _, tool in pairs] + [stray]
    stats = object_frame_tool_offsets(objects, tools, yaw_period_deg=180.0)
    assert stats is not None
    assert stats.count == 5
    assert stats.mean_xy_m == pytest.approx(local, abs=1e-12)
    assert stats.std_xy_m == pytest.approx((0.0, 0.0), abs=1e-12)
    assert max(stats.base_frame_std_xy_m) > 0.005
    assert -90.0 < ((stats.reference_yaw_deg + 90.0) % 180.0) - 90.0 < 52.0


def test_object_frame_offsets_reject_arm_mismatch_and_handle_no_pairs() -> None:
    obj, tool = _object_and_tool(0, 0.8, 0.1, 10.0, (0.01, 0.0))
    assert object_frame_tool_offsets([obj], [], yaw_period_deg=180.0) is None
    with pytest.raises(ValueError, match="arm"):
        object_frame_tool_offsets([obj], [{**tool, "arm": "left"}], yaw_period_deg=180.0)


def test_object_frame_reference_yaw_is_centred_so_offset_sign_is_stable() -> None:
    local = (0.025, 0.006)
    pairs = [_object_and_tool(ep, 0.8, 0.1, yaw, local) for ep, yaw in enumerate([-1.5, -1.0, 0.5])]
    stats = object_frame_tool_offsets(
        [o for o, _ in pairs], [t for _, t in pairs], yaw_period_deg=180.0
    )
    assert stats is not None
    assert -90.0 < stats.reference_yaw_deg <= 90.0
    assert stats.reference_yaw_deg == pytest.approx(0.0, abs=1.0)
    assert stats.mean_xy_m == pytest.approx(local, abs=1e-12)
