"""Chart theming utilities for DashMat."""


def apply_chart_theme(fig, theme):
    """Apply dark or light theme to a Plotly figure."""
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
