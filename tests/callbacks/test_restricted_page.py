from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def restricted_module():
    import app  # noqa: F401
    import pages.restricted as restricted

    return restricted


def test_decode_target_unquotes_spaces(restricted_module):
    assert restricted_module._decode_target("?target=Analytics%20Tool") == "Analytics Tool"


def test_decode_target_falls_back_when_missing(restricted_module):
    assert restricted_module._decode_target("?foo=bar") == "this page"


def test_update_restricted_message_uses_decoded_target(restricted_module):
    text = restricted_module.update_restricted_message("?target=Portfolio%20Optimization")
    assert text == "Your account does not have access to Portfolio Optimization."
