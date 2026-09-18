"""docs/status.md 是公开 API 与 frontend intrinsic 的状态事实来源。"""

import re
from pathlib import Path

import tila
from tila.frontend import INTRINSIC_NAMES


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs" / "status.md"
VALID_STATES = {"Implemented", "Partial", "Designed", "Deferred", "Removed"}


def _inventory(text: str, name: str) -> set[str]:
    match = re.search(rf"<!-- {re.escape(name)}: ([^>]+) -->", text)
    assert match is not None, f"missing {name} inventory in docs/status.md"
    return {item.strip() for item in match.group(1).split(",") if item.strip()}


def test_status_doc_tracks_public_api_exactly():
    text = STATUS.read_text(encoding="utf-8")
    assert _inventory(text, "public-api") == set(tila.__all__)


def test_status_doc_tracks_frontend_intrinsics_exactly():
    text = STATUS.read_text(encoding="utf-8")
    assert _inventory(text, "frontend-intrinsics") == set(INTRINSIC_NAMES)


def test_status_tables_only_use_defined_states():
    text = STATUS.read_text(encoding="utf-8")
    rows = [line for line in text.splitlines()
            if line.startswith("|") and not line.startswith("|---")]
    state_cells = []
    for row in rows:
        cells = [cell.strip().strip("`") for cell in row.strip("|").split("|")]
        if len(cells) >= 2 and cells[1] in VALID_STATES:
            state_cells.append(cells[1])
    assert state_cells, "status.md contains no machine-recognizable capability rows"
    assert VALID_STATES <= set(state_cells), "every state must be represented and auditable"
