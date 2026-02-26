"""Chart theming utilities for DashMat."""


def apply_chart_theme(fig, theme):
    """Apply dark or light theme to a Plotly figure."""
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    hoverlabel = (
        {
            "bgcolor": "#111827",
            "font": {"color": "#F8FAFC"},
            "bordercolor": "#64748B",
        }
        if theme == "dark"
        else {
            "bgcolor": "#FFFFFF",
            "font": {"color": "#111827"},
            "bordercolor": "#CBD5E1",
        }
    )
    margin = {}
    if fig.layout and fig.layout.margin:
        margin = fig.layout.margin.to_plotly_json()
    if margin.get("r", 0) < 160:
        margin["r"] = 160
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.01,
        title_y=0.98,
        title_xanchor="left",
        title_yanchor="top",
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        hoverlabel=hoverlabel,
        margin=margin,
    )
    return fig
