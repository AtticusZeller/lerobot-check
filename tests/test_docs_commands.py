import re
import shlex
from pathlib import Path

import pytest

from lerobot_check.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
LAYERED_COMMANDS = ("suggest", "collection-plan", "mask-template", "mask-overlays")


def _documented_commands() -> list[tuple[str, list[str]]]:
    commands = []
    for doc in ("README.md", "docs/collection-plan.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for block in re.findall(r"```(?:bash|sh)\n(.*?)```", text, flags=re.S):
            for line in block.replace("\\\n", " ").splitlines():
                tokens = shlex.split(line, comments=True)
                if "lerobot-check" not in tokens:
                    continue
                argv = tokens[tokens.index("lerobot-check") + 1 :]
                if argv and argv[0] in LAYERED_COMMANDS:
                    commands.append((doc, argv))
    return commands


DOCUMENTED = _documented_commands()


def test_documented_layer_commands_are_found():
    assert {argv[0] for _, argv in DOCUMENTED} == set(LAYERED_COMMANDS)


@pytest.mark.parametrize(("doc", "argv"), DOCUMENTED, ids=[" ".join(a[:2]) for _, a in DOCUMENTED])
def test_documented_command_parses_with_shipped_flags(doc, argv):
    args = build_parser().parse_args(argv)
    if argv[0] in ("suggest", "mask-template") or argv[:2] == ["collection-plan", "generate"]:
        assert args.layer == "object"
