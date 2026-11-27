import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot_check.collection_progress import (
    PlanBusyError,
    import_progress,
    load_plan,
    next_pending,
    plan_session,
    validate_capture,
)

EXAMPLE = Path(__file__).parent / "fixtures" / "collection-plan.example.json"


@pytest.fixture
def plan_file(tmp_path):
    return Path(shutil.copy(EXAMPLE, tmp_path / "plan.json"))


def test_stdlib_module_is_directly_reusable(plan_file, tmp_path):
    module = Path(__file__).parents[1] / "src/lerobot_check/collection_progress.py"
    shutil.copyfile(module, tmp_path / "collection_progress.py")
    script = tmp_path / "capture.py"
    script.write_text(
        "from collection_progress import plan_session, next_pending\n"
        "from pathlib import Path\n"
        "with plan_session(Path('plan.json')) as session:\n"
        "    session.mark_collected(next_pending(session.read())['id'], 100)\n"
    )
    subprocess.run([sys.executable, "-S", str(script)], cwd=tmp_path, check=True)
    assert load_plan(plan_file)["rounds"][0]["episode_index"] == 100


def test_session_lock_and_replacement_failure_keep_old_state(plan_file, monkeypatch):
    before = plan_file.read_bytes()
    with plan_session(plan_file) as session:
        with pytest.raises(PlanBusyError), plan_session(plan_file):
            pytest.fail("second writer acquired the same lock")

        def interrupt(_source, _dest):
            assert plan_file.read_bytes() == before
            assert load_plan(plan_file)["rounds"][0]["status"] == "pending"
            raise OSError("interrupted before rename")

        with monkeypatch.context() as patch:
            patch.setattr(os, "replace", interrupt)
            with pytest.raises(OSError, match="interrupted"):
                session.mark_skipped(1)
        assert plan_file.read_bytes() == before
        assert set(p.name for p in plan_file.parent.iterdir()) == {"plan.json", "plan.json.lock"}
        session.mark_skipped(1)
        pending = next_pending(session.read())
        assert pending is not None and pending["id"] == 2
    with pytest.raises(RuntimeError, match="closed"):
        session.mark_skipped(2)
    with plan_session(plan_file) as fresh:
        assert fresh.read()["rounds"][0]["status"] == "skipped"


def test_progress_requires_real_unique_append_index(plan_file):
    with plan_session(plan_file) as session:
        with pytest.raises(ValueError, match="new and unique"):
            session.mark_collected(1, 99)
        assert session.mark_collected(1, 100) == session.mark_collected(1, 100)
        with pytest.raises(ValueError, match="new and unique"):
            session.mark_collected(2, 100)
        with pytest.raises(ValueError, match="already collected"):
            session.mark_skipped(1)
    assert load_plan(plan_file)["generation"] == 1
    pending = next_pending(load_plan(plan_file))
    assert pending is not None and pending["id"] == 2


def test_import_copies_only_progress_and_rejects_conflict(plan_file, tmp_path):
    remote = Path(shutil.copy(plan_file, tmp_path / "returned.json"))
    with plan_session(remote) as session:
        session.mark_skipped(2)
    old = load_plan(plan_file)
    merged = import_progress(plan_file, remote)
    assert merged["rounds"][1]["status"] == "skipped"
    assert merged["rounds"][0] == old["rounds"][0]
    assert merged["rounds"][1]["targets"] == old["rounds"][1]["targets"]
    assert merged == import_progress(plan_file, remote)
    with plan_session(remote) as session:
        session.mark_collected(1, 101)
    with plan_session(plan_file) as session:
        session.mark_collected(1, 100)
    with pytest.raises(ValueError, match="Conflicting progress"):
        import_progress(plan_file, remote)
    assert load_plan(plan_file)["rounds"][0]["episode_index"] == 100
    with plan_session(remote) as session:
        newer = session.read()
        newer["generation"] += 1
        session.write(newer)
    with pytest.raises(ValueError, match="generation"):
        import_progress(plan_file, remote)


def test_capture_identity_rejects_camera_or_dataset_mismatch(plan_file):
    plan = load_plan(plan_file)

    def validate(
        *,
        repo_id="org/demo",
        revision="local",
        episode_count=100,
        camera="observation.images.head",
        serial="0000123",
        image_size=(640, 480),
    ):
        validate_capture(
            plan,
            repo_id=repo_id,
            revision=revision,
            episode_count=episode_count,
            camera=camera,
            serial=serial,
            image_size=image_size,
        )

    validate()
    with pytest.raises(ValueError, match="live head camera"):
        validate(serial="123")
    with pytest.raises(ValueError, match="live head camera"):
        validate(image_size=(480, 640))
    with pytest.raises(ValueError, match="live head camera"):
        validate(camera="left")
    with pytest.raises(ValueError, match="dataset identity"):
        validate(repo_id="other/name")
    with pytest.raises(ValueError, match="dataset identity"):
        validate(revision="other")
    with pytest.raises(ValueError, match="episode count"):
        validate(episode_count=99)
    plan["camera_serial"] = None
    with pytest.raises(ValueError, match="live head camera"):
        validate()


@pytest.mark.parametrize(
    "raw", ["{", '{"version": 1, "version": 1}', '{"number": NaN}', '{"big": 1e3000}']
)
def test_corrupt_plan_refused(tmp_path, raw):
    file = tmp_path / "bad.json"
    file.write_text(raw)
    with pytest.raises(ValueError, match="Invalid collection plan"):
        load_plan(file)


def test_plan_layer_validation(plan_file):
    from lerobot_check.collection_progress import validate_plan

    plan = load_plan(plan_file)
    assert "layer" not in plan
    validate_plan({**plan, "layer": "object"})
    validate_plan({**plan, "layer": "tool"})
    with pytest.raises(ValueError, match="layer"):
        validate_plan({**plan, "layer": "unknown"})
