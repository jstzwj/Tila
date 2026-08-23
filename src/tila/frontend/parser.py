"""parse：ast.parse 包装（SyntaxError → E11）。"""

from __future__ import annotations

import ast

from ..diagnostics import Loc, TilaError, err


def parse(source: str):
    """解析 .tila 源码（内容必须是语法合法的 Python）；SyntaxError 包装成 E11。"""
    try:
        return ast.parse(source)
    except SyntaxError as e:
        loc = Loc(e.lineno or 1, e.offset or 1)
        raise err(
            loc,
            "E11",
            f"source is not valid Python syntax: {e.msg}",
            "Tila v0.1 is a decidable subset of Python; parsing uses ast.parse directly",
        ) from e
