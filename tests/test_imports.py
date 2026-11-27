import subprocess
import sys


def test_alignment_import_does_not_load_relabel() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import lerobot_check.alignment

assert 'lerobot_check.relabel' not in sys.modules
assert 'lerobot.datasets.io_utils' not in sys.modules

import lerobot_check

assert 'relabel_dataset' in dir(lerobot_check)
from lerobot_check import relabel_dataset
from lerobot_check.relabel import relabel_dataset as implementation
assert relabel_dataset is implementation
""",
        ],
        check=True,
    )
