import plotly.graph_objects as go

from utils.charting import apply_chart_theme


def test_apply_chart_theme_sets_dark_hoverlabel_colors():
    fig = go.Figure()

    apply_chart_theme(fig, "dark")

    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.layout.hoverlabel.bgcolor == "#25262b"
    assert fig.layout.hoverlabel.bordercolor == "#5c5f66"
    assert fig.layout.hoverlabel.font.color == "#f8f9fa"
    assert fig.layout.hoverlabel.namelength == -1


def test_apply_chart_theme_sets_light_hoverlabel_colors():
    fig = go.Figure()

    apply_chart_theme(fig, "light")

    assert fig.layout.hoverlabel.bgcolor == "#ffffff"
    assert fig.layout.hoverlabel.bordercolor == "#ced4da"
    assert fig.layout.hoverlabel.font.color == "#1f2933"
    assert fig.layout.hoverlabel.namelength == -1
