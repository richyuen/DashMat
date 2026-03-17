from utils.ag_grid import literal_field_dash_grid_options


def test_literal_field_dash_grid_options_injects_clipboard_processor():
    opts = literal_field_dash_grid_options({"animateRows": True})

    assert opts["animateRows"] is True
    assert opts["suppressFieldDotNotation"] is True
    assert opts["processCellForClipboard"] == {"function": "dashmatProcessCellForClipboard(params)"}
