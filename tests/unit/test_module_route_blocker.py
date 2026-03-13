from utils import module_route_blocker as blocker


def test_is_module_route_scopes_only_dashmat_modules():
    assert blocker.is_module_route("/analyticstool")
    assert blocker.is_module_route("/portopt/")
    assert blocker.is_module_route("/regression?x=1")
    assert not blocker.is_module_route("/")
    assert not blocker.is_module_route("/restricted")
    assert not blocker.is_module_route("/other")


def test_route_blocker_shell_style_hides_when_inactive():
    assert blocker.route_blocker_shell_style(False) == {"display": "none"}
    assert blocker.route_blocker_shell_style(True) == {
        "position": "fixed",
        "inset": 0,
        "zIndex": 2400,
    }


def test_build_module_route_blocker_components_exposes_expected_ids():
    components = blocker.build_module_route_blocker_components()
    ids = [getattr(component, "id", None) for component in components]
    assert "dashmat-route-blocker-store" in ids
    assert "at-route-ready-store" in ids
    assert "po-route-ready-store" in ids
    assert "reg-route-ready-store" in ids
    assert "dashmat-route-blocker-shell" in ids
