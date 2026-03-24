from pathlib import Path
import re

_FIXTURE = Path(__file__).parent / "fixtures" / "upstream_add_trace.txt"

def test_upstream_add_trace_fixture_exists_and_has_header():
    text = _FIXTURE.read_text()
    lines = text.splitlines()
    first = lines[0]
    assert first.startswith("// target=")
    assert re.search(r"block_dim=\d+", first)
    # Guard against a truncated fixture: must have substantial content and
    # contain at least one disassembly opcode line.
    assert len(lines) > 5
    assert any("." in line and not line.startswith("//") for line in lines)
