from utils.dashmat_welcome_modal import (
    _ANCHORED_GROWING_MODAL_STYLES,
    _ANCHORED_GROWING_MODAL_Y_OFFSET,
    _UPWARD_MULTISELECT_COMBOBOX_PROPS,
    build_db_add_modal,
    build_sheet_select_modal,
    build_underlying_add_modal,
)


def test_db_add_modal_uses_anchored_scrolling_layout():
    modal = build_db_add_modal("at")
    series_select = modal.children[1]

    assert modal.centered is False
    assert modal.yOffset == _ANCHORED_GROWING_MODAL_Y_OFFSET
    assert modal.styles == _ANCHORED_GROWING_MODAL_STYLES
    assert modal.size == "820px"
    assert series_select.maxDropdownHeight == 380
    assert series_select.comboboxProps == _UPWARD_MULTISELECT_COMBOBOX_PROPS


def test_underlying_add_modal_uses_upward_multiselects():
    modal = build_underlying_add_modal("at", "test-key")
    body = modal.children[1]
    type_select = body.children[0].children[1]
    desc_select = body.children[1]

    assert modal.centered is False
    assert modal.yOffset == _ANCHORED_GROWING_MODAL_Y_OFFSET
    assert modal.styles == _ANCHORED_GROWING_MODAL_STYLES
    assert type_select.comboboxProps == _UPWARD_MULTISELECT_COMBOBOX_PROPS
    assert desc_select.comboboxProps == _UPWARD_MULTISELECT_COMBOBOX_PROPS


def test_sheet_select_modal_uses_anchored_multiselect():
    modal = build_sheet_select_modal("at")
    sheet_select = modal.children[1]

    assert modal.centered is False
    assert modal.yOffset == _ANCHORED_GROWING_MODAL_Y_OFFSET
    assert modal.styles == _ANCHORED_GROWING_MODAL_STYLES
    assert sheet_select.comboboxProps == _UPWARD_MULTISELECT_COMBOBOX_PROPS
