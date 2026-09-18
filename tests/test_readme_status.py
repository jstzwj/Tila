"""README 只做已验证入口，不复制易漂移或超前的状态承诺。"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_links_authoritative_status_and_plan():
    assert "[docs/status.md](docs/status.md)" in README
    assert "[plan.md](plan.md)" in README
    assert "当前能力的唯一状态清单" in README


def test_readme_does_not_hardcode_test_or_source_counts():
    assert not re.search(r"\d+\s*项测试", README)
    assert not re.search(r"约\s*\d+\s*行", README)
    assert "零 skipped" in README


def test_readme_marks_non_closed_paths_as_limits():
    limits = README.split("## 当前限制", 1)[1]
    for term in ("Triton/CUDA", "Ptr", "FP8", "TypeVar", "SMT", "race"):
        assert term in limits


def test_quickstart_commands_reference_existing_files():
    for name in ("add_kernel.py", "matmul.py", "self_attention.py",
                 "fused_attention.py"):
        assert (ROOT / "examples" / name).is_file()
        assert f"python examples/{name}" in README
