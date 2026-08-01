import subprocess, shutil
from pathlib import Path
import pytest
from scripts._compare import theme


def _validate(hexes, mode, surface):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = theme.VALIDATOR_PATH  # vendored tests/fixtures/validate_palette.js (set in Task 8 Step 3)
    out = subprocess.run([node, str(script), ",".join(hexes), "--mode", mode, "--surface", surface],
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def test_engine_pair_validates_light():
    rc, log = _validate(list(theme.ENGINE.values()), "light", theme.SURFACE["light"]["bg"])
    assert rc == 0, log


def test_engine_pair_validates_dark():
    rc, log = _validate(list(theme.ENGINE.values()), "dark", theme.SURFACE["dark"]["bg"])
    assert rc == 0, log


def test_status_is_glyph_plus_label():
    for k, v in theme.STATUS.items():
        assert v["glyph"] and v["label"]      # never color-alone


def test_engine_slots_are_fixed_order():
    assert list(theme.ENGINE.keys()) == ["candidate", "reference"]
    assert theme.ENGINE["candidate"] != theme.ENGINE["reference"]


def test_engine_colors_distinct_from_status_colors():
    status_hexes = {v["color"].lower() for v in theme.STATUS.values()}
    for hexval in theme.ENGINE.values():
        assert hexval.lower() not in status_hexes


def test_surface_has_required_keys():
    for mode in ("light", "dark"):
        for key in ("bg", "ink", "muted", "line", "card"):
            assert key in theme.SURFACE[mode], f"{mode}.{key} missing"


def test_css_vars_emits_root_block():
    for mode in ("light", "dark"):
        css = theme.css_vars(mode)
        assert css.strip().startswith(":root")
        assert "--bg" in css
        assert theme.SURFACE[mode]["bg"] in css
        assert theme.ENGINE["candidate"] in css
        assert theme.STATUS["within_tol"]["color"] in css


def test_css_vars_rejects_unknown_mode():
    with pytest.raises(ValueError):
        theme.css_vars("sepia")


def test_validator_path_exists():
    assert Path(theme.VALIDATOR_PATH).is_file()
