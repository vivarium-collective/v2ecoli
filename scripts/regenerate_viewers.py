#!/usr/bin/env python
"""Regenerate the composite viewers hub (docs/viewers/).

For each composite in docs/viewers/viewers.json this resolves the composite to a
cached state JSON, pre-renders a bigraph-viz SVG and a self-contained
bigraph-viz2 HTML snippet, copies the shared bigraph-loom static bundle, and
regenerates the hub index.html. Unresolvable composites are skipped with a
warning (never fatal).

Run locally (the ParCa cache must be on disk so heavy composites resolve):
    PYTHONPATH=. .venv/bin/python scripts/regenerate_viewers.py
"""
import json
import shutil
import sys
from dataclasses import dataclass
from html import escape as _esc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VIEWERS_DIR = REPO_ROOT / "docs" / "viewers"


@dataclass
class Entry:
    slug: str
    id: str
    title: str
    blurb: str = ""


def load_manifest(path: Path) -> list[Entry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Entry(slug=e["slug"], id=e["id"], title=e["title"],
                  blurb=e.get("blurb", "")) for e in raw]


def trim_state_for_view(obj, *, max_list: int = 8):
    """Shrink a resolved composite state for VIEWING.

    The bigraph STRUCTURE the viewers draw (stores, processes, wiring, schemas,
    describe() docs) lives in dicts and short path lists. The bulk weight is
    long molecule-data arrays — bulk counts (~16k), unique-molecule instance
    lists (active_ribosome ~12k, promoter ~5k, …) — which loom/viz2/viz never
    render. Capping every long list (numpy arrays included) keeps the cached
    state small (5.5 MB -> a few hundred KB) without touching structure.
    """
    if isinstance(obj, dict):
        return {k: trim_state_for_view(v, max_list=max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # NOTE: this caps EVERY list, including any structural list (e.g. a
        # process with >max_list ports expressed as a list). All current
        # composites are safe — their structural lists are short wiring paths
        # (<=4) — but a future composite with a long list in schema position
        # would be silently truncated; bump max_list or special-case if so.
        return [trim_state_for_view(v, max_list=max_list) for v in obj[:max_list]]
    # numpy (and other 1-D array-likes): cap then recurse on the python list.
    tolist = getattr(obj, "tolist", None)
    if callable(tolist) and getattr(obj, "ndim", 0) >= 1:
        try:
            return trim_state_for_view(obj[:max_list].tolist(), max_list=max_list)
        except Exception:  # noqa: BLE001 — fall through to leave the value as-is
            pass
    return obj


def write_state(state: dict, slug: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"{slug}.state.json"
    # Serialize EXACTLY as the dashboard's /api/composite-state does, via its
    # _json_body: _json_default turns numpy arrays/scalars, Path, and sets into
    # JSON-native values (composite states are full of numpy bulk-count arrays),
    # and allow_nan=False + the _json_sanitize fallback replaces inf/nan with
    # null. Plain json.dumps chokes on the ndarrays. loom's ?stateUrl= reader
    # expects the {"state": <bigraph-state>} wrapper.
    from vivarium_dashboard.server import _json_body
    out.write_bytes(_json_body({"state": state}))
    return out


def loom_url(entry: Entry, *, has_view: bool) -> str:
    url = (f"loom/index.html?static=1&id={entry.id}"
           f"&stateUrl=../data/{entry.slug}.state.json")
    if has_view:
        url += f"&viewUrl=../data/{entry.slug}.view.json"
    return url


def hub_html(rows: list[dict]) -> str:
    """rows: [{entry: Entry, has_view: bool, has_viz2: bool, has_svg: bool}]."""
    cards = []
    for r in rows:
        e = r["entry"]
        buttons = [f'<a class="btn" href="{loom_url(e, has_view=r["has_view"])}">Loom</a>']
        if r["has_viz2"]:
            buttons.append(f'<a class="btn" href="viz2/{e.slug}.html">Viz2</a>')
        if r["has_svg"]:
            buttons.append(f'<a class="btn" href="img/{e.slug}.svg">Viz</a>')
        cards.append(
            '<div class="card">'
            f'<h2>{_esc(e.title)}</h2>'
            f'<p class="blurb">{_esc(e.blurb)}</p>'
            f'<p class="id"><code>{_esc(e.id)}</code></p>'
            f'<div class="btns">{"".join(buttons)}</div>'
            '</div>')
    return _PAGE_TEMPLATE.replace("{{CARDS}}", "\n".join(cards))


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>v2ecoli — composite viewers</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#f7f7f8;color:#1b1b1f}
 header{padding:24px 32px;background:#fff;border-bottom:1px solid #e3e3e8}
 h1{margin:0;font-size:1.4rem} .sub{color:#666;margin:6px 0 0;font-size:.9rem}
 main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px;padding:24px 32px}
 .card{background:#fff;border:1px solid #e3e3e8;border-radius:10px;padding:16px 18px}
 .card h2{margin:0 0 4px;font-size:1.05rem} .blurb{margin:0 0 8px;color:#444;font-size:.9rem}
 .id{margin:0 0 12px} .id code{font-size:.75rem;color:#777}
 .btns{display:flex;gap:8px} .btn{display:inline-block;padding:5px 12px;border-radius:6px;
   background:#1f6feb;color:#fff;text-decoration:none;font-size:.85rem}
 .btn:hover{background:#1a5fd0}
</style></head>
<body>
<header><h1>v2ecoli composite viewers</h1>
<p class="sub">Each composite, viewable in bigraph-loom (interactive), bigraph-viz2 (interactive, light), and bigraph-viz (static). Cached states regenerated by scripts/regenerate_viewers.py.</p></header>
<main>
{{CARDS}}
</main>
</body></html>
"""


def resolve_state_via_dashboard(spec_id: str) -> dict | None:
    """Resolve a composite to its loom state dict by reusing the dashboard's
    pure resolver. Returns None on failure, after surfacing WHY."""
    import vivarium_dashboard.server as srv
    # This script is always run standalone; point the dashboard's pure resolver at this workspace.
    srv.WORKSPACE = REPO_ROOT
    data = srv._composite_resolve_data(spec_id)
    if data and isinstance(data.get("state"), dict):
        return data["state"]
    # _composite_resolve_data swallows the real error and returns None. Re-run
    # the generator directly to surface it — the usual culprit is a stale ParCa
    # cache (e.g. a CountsDeriver config missing 'tf_ids'), fixed by rebuilding:
    #   PYTHONPATH=. .venv/bin/python scripts/build_cache.py
    _log_resolve_failure(spec_id)
    return None


def _log_resolve_failure(spec_id: str) -> None:
    """Print the underlying reason a composite failed to resolve (the dashboard
    resolver hides it behind a None return)."""
    try:
        from pbg_superpowers.composite_generator import (
            _REGISTRY, build_generator, discover_generators,
        )
        if spec_id not in _REGISTRY:
            discover_generators()
        entry = _REGISTRY.get(spec_id)
        if entry is None:
            reason = "composite is not registered (check the id in viewers.json)"
        else:
            build_generator(entry, overrides={})
            reason = "resolver returned None but the generator built — unexpected"
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
    print(f"  ! skipped {spec_id}: {reason}\n"
          f"    (if this is a stale cache, run: "
          f"PYTHONPATH=. .venv/bin/python scripts/build_cache.py)",
          file=sys.stderr)


def render_viz_svg(state: dict, slug: str, img_dir: Path) -> Path | None:
    """Best-effort bigraph-viz static SVG. Returns the path or None on failure."""
    try:
        from bigraph_viz import plot_bigraph
        img_dir.mkdir(parents=True, exist_ok=True)
        plot_bigraph(state, out_dir=str(img_dir), filename=slug,
                     file_format="svg", show_compiled_state=False)
        # plot_bigraph also writes the intermediate Graphviz DOT source as a
        # bare <slug> file (no extension) alongside <slug>.svg — drop it.
        dot_src = img_dir / slug
        if dot_src.is_file():
            dot_src.unlink()
        out = img_dir / f"{slug}.svg"
        return out if out.is_file() else None
    except Exception as exc:  # noqa: BLE001
        print(f"  ! viz SVG skipped for {slug}: {exc}", file=sys.stderr)
        return None


def render_viz2_html(state: dict, slug: str, viz2_dir: Path) -> Path | None:
    """Best-effort self-contained bigraph-viz2 snippet. Returns path or None."""
    try:
        from bigraph_viz2 import emit_html
        viz2_dir.mkdir(parents=True, exist_ok=True)
        html = emit_html(state, height="100vh")
        out = viz2_dir / f"{slug}.html"
        out.write_text(html, encoding="utf-8")
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  ! viz2 skipped for {slug}: {exc}", file=sys.stderr)
        return None


def build_rows(entries, *, viewers_dir, resolve, render_svg, render_viz2):
    data_dir = viewers_dir / "data"
    img_dir = viewers_dir / "img"
    viz2_dir = viewers_dir / "viz2"
    rows = []
    for e in entries:
        print(f"- {e.slug} ({e.id})")
        try:
            state = resolve(e.id)
            if state is None:
                continue  # resolve() already logged why
            state = trim_state_for_view(state)  # drop bulk data arrays the viewers don't draw
            write_state(state, e.slug, data_dir)
            svg = render_svg(state, e.slug, img_dir)
            viz2 = render_viz2(state, e.slug, viz2_dir)
        except Exception as exc:  # noqa: BLE001 — one bad composite must not abort the run
            print(f"  ! skipped {e.slug}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        rows.append({
            "entry": e,
            "has_view": (data_dir / f"{e.slug}.view.json").is_file(),
            "has_svg": svg is not None,
            "has_viz2": viz2 is not None,
        })
    return rows


def loom_dist_dir() -> Path:
    """Locate the installed bigraph-loom static build (_dist/)."""
    import bigraph_loom
    return Path(bigraph_loom.__file__).resolve().parent / "_dist"


def copy_loom_bundle(dest: Path, *, src: Path | None = None) -> None:
    src = src or loom_dist_dir()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    # A read-only viewer never needs JS source maps (~8 MB, half the bundle) —
    # strip them, matching scripts/publish_dashboard.sh.
    for m in dest.rglob("*.map"):
        m.unlink()


def main() -> int:
    entries = load_manifest(VIEWERS_DIR / "viewers.json")
    print(f"Regenerating {len(entries)} composite viewers into {VIEWERS_DIR}")
    rows = build_rows(entries, viewers_dir=VIEWERS_DIR,
                      resolve=resolve_state_via_dashboard,
                      render_svg=render_viz_svg, render_viz2=render_viz2_html)
    copy_loom_bundle(VIEWERS_DIR / "loom")
    (VIEWERS_DIR / "index.html").write_text(hub_html(rows), encoding="utf-8")
    print(f"Done: {len(rows)}/{len(entries)} composites in the hub "
          f"({VIEWERS_DIR / 'index.html'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
