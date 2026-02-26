from __future__ import annotations

import plotly.graph_objects as go

from utils.charting import apply_chart_theme


def test_apply_chart_theme_sets_dark_hoverlabel_contrast():
    fig = go.Figure()
    themed = apply_chart_theme(fig, "dark")

    assert themed.layout.hoverlabel.bgcolor == "#111827"
    assert themed.layout.hoverlabel.font.color == "#F8FAFC"
    assert themed.layout.hoverlabel.bordercolor == "#64748B"


def test_apply_chart_theme_sets_light_hoverlabel_contrast():
    fig = go.Figure()
    themed = apply_chart_theme(fig, "light")

    assert themed.layout.hoverlabel.bgcolor == "#FFFFFF"
    assert themed.layout.hoverlabel.font.color == "#111827"
    assert themed.layout.hoverlabel.bordercolor == "#CBD5E1"
