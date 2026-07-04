# Pre-resolved composite states (dashboard explorer)

Each `<composite-id>.json` here is a **pre-resolved composite state** the
read-only dashboard publish step ships verbatim as
`api/composite-state/<id>.json` and marks navigable (`has_wiring=True`) — even
if the composite can't resolve at publish time.

`vivarium_workbench.publish` checks this directory first; a committed file wins
over live resolution. (See the publish loop in vivarium-workbench's
`publish.py`.)

## Why this exists

Some composites can't resolve in CI/publish because they need on-disk inputs.
The full **`baseline`** composite (the 55-process whole-cell model) calls
`load_cache_bundle("out/cache")`, which needs the ParCa cache — absent at
publish time. Without a committed state it would resolve to `has_wiring=False`
and the dashboard would hide its Explore button.

## Regenerating `v2ecoli.composites.baseline.baseline.json`

This is a **manual artifact** — refresh it when the baseline model topology
changes. On a machine with a valid ParCa cache (`out/cache/`, e.g. the lab
mini — full ParCa build is ~2.5 min):

```python
# resolve the baseline, prune heavy arrays, serialize via the server serializer
import os, sys, json, math
os.environ.setdefault("POLARS_MAX_THREADS", "1")
from pathlib import Path
WS = Path("/path/to/v2ecoli"); sys.path.insert(0, str(WS))
from vivarium_workbench import server
from vivarium_workbench.lib._root import set_workspace_root
from vivarium_workbench.server import _json_default
server.WORKSPACE = WS; set_workspace_root(WS)

CID = "v2ecoli.composites.baseline.baseline"
data = server._composite_resolve_data(CID)            # needs out/cache

def prune(o):                                          # truncate heavy arrays; keep topology
    tn = type(o).__name__
    if tn == "ndarray":
        n = int(getattr(o, "size", 0))
        return f"<ndarray shape={tuple(o.shape)} dtype={o.dtype}>" if n > 16 else prune(o.tolist())
    if tn.startswith("float"): v = float(o); return v if math.isfinite(v) else None
    if tn.startswith(("int", "uint")): return int(o)
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, (list, tuple)):
        return ([prune(x) for x in o[:16]] + [f"...(+{len(o)-16} more)"]) if len(o) > 16 else [prune(x) for x in o]
    if isinstance(o, dict): return {k: prune(v) for k, v in o.items()}
    if isinstance(o, str) and len(o) > 400: return o[:400] + "…"
    return o

data["state"] = prune(data.get("state", {}))
Path(f"reports/composite-state/{CID}.json").write_text(
    json.dumps(data, default=_json_default, allow_nan=False), encoding="utf-8")
```

The pruning truncates the multi-MB bulk store and other large arrays (the
explorer only needs the **topology** — process/store nodes + wiring — not the
data values), keeping the committed file a few MB.

## Regenerating `v2ecoli.composites.biological.biological.json`

The **`biological`** composite is `baseline` with a pure path-remap of the store
hierarchy (`v2ecoli.composites._remap.remap_cell_state` — same simulation, just
relabeled). Its `baseline()` call needs the ParCa cache too, so it likewise
can't resolve at publish time and ships a committed state.

Because it's a pure relabel, its committed state is **derived from the baseline
file** — no ParCa cache needed. Refresh it whenever the baseline file changes:

```python
import json, importlib.util
spec = importlib.util.spec_from_file_location("_remap", "v2ecoli/composites/_remap.py")
rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)

d = json.load(open("reports/composite-state/v2ecoli.composites.baseline.baseline.json"))
d["id"] = "v2ecoli.composites.biological.biological"
d["name"] = "biological"
d["module"] = "v2ecoli.composites.biological"
d["description"] = "Biologically-organized whole-cell E. coli model. …"  # match the decorator
agents = d["state"]["agents"]
for aid, cell in list(agents.items()):
    agents[aid] = rm.remap_cell_state(cell)       # exactly what biological() does
json.dump(d, open("reports/composite-state/v2ecoli.composites.biological.biological.json", "w"),
          allow_nan=False)
```
