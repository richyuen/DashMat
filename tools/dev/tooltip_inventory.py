"""Inventory interactive controls and tooltip coverage for DashMat pages."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ui_tooltips import has_custom_tooltip, is_interactive_component_name, tooltip_source


TARGET_FILES = [
    Path("pages/analyticstool.py"),
    Path("pages/regression.py"),
    Path("pages/portopt.py"),
]


@dataclass(frozen=True)
class ControlRef:
    page: str
    namespace: str
    component: str
    control_id: str
    line: int


class _InteractiveIdVisitor(ast.NodeVisitor):
    def __init__(self, page_name: str):
        self.page_name = page_name
        self.controls: list[ControlRef] = []

    def visit_Call(self, node: ast.Call):
        namespace, component_name = self._component_name(node.func)
        if namespace and component_name and is_interactive_component_name(namespace, component_name):
            control_id = self._string_keyword(node, "id")
            if control_id:
                self.controls.append(
                    ControlRef(
                        page=self.page_name,
                        namespace=namespace,
                        component=component_name,
                        control_id=control_id,
                        line=int(getattr(node, "lineno", 0) or 0),
                    )
                )
        self.generic_visit(node)

    @staticmethod
    def _component_name(func_node: ast.AST) -> tuple[str | None, str | None]:
        if not isinstance(func_node, ast.Attribute):
            return None, None
        if not isinstance(func_node.value, ast.Name):
            return None, None
        namespace = str(func_node.value.id or "").strip()
        if namespace not in {"dmc", "dcc", "html"}:
            return None, None
        component_name = str(func_node.attr or "").strip()
        if not component_name:
            return None, None
        return namespace, component_name

    @staticmethod
    def _string_keyword(node: ast.Call, key: str) -> str | None:
        for kw in node.keywords:
            if str(kw.arg or "") != key:
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.strip():
                return value.value.strip()
        return None


def _collect_controls(path: Path) -> list[ControlRef]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _InteractiveIdVisitor(page_name=path.stem)
    visitor.visit(tree)
    return visitor.controls


def main():
    all_controls: list[ControlRef] = []
    for path in TARGET_FILES:
        if not path.exists():
            continue
        all_controls.extend(_collect_controls(path))

    # De-duplicate IDs while preserving first-seen context.
    dedup_map: dict[str, ControlRef] = {}
    for item in all_controls:
        dedup_map.setdefault(item.control_id, item)
    controls = list(dedup_map.values())
    controls.sort(key=lambda item: (item.page, item.line, item.control_id))

    total = len(controls)
    custom = [item for item in controls if has_custom_tooltip(item.control_id)]
    fallback = [item for item in controls if not has_custom_tooltip(item.control_id)]
    custom_ratio = (len(custom) / total * 100.0) if total else 0.0

    print("DashMat Tooltip Inventory")
    print("=" * 26)
    print(f"Target pages: {', '.join(str(path) for path in TARGET_FILES)}")
    print(f"Interactive controls (unique IDs): {total}")
    print(f"Custom tooltip coverage: {len(custom)} / {total} ({custom_ratio:.1f}%)")
    print(f"Fallback tooltip IDs: {len(fallback)}")

    if fallback:
        print("\nFallback IDs (review candidates):")
        for item in fallback:
            source = tooltip_source(item.control_id)
            print(f"- [{item.page}:{item.line}] {item.control_id} ({item.namespace}.{item.component}, source={source})")


if __name__ == "__main__":
    main()
