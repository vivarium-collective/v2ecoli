# Converting vivarium-1.0 processes to process-bigraph

v2ecoli runs on [process-bigraph](https://github.com/vivarium-collective/process-bigraph),
but most upstream biology was originally written as
[vivarium-1.0](https://github.com/vivarium-collective/vivarium-core) processes
(classes with `ports_schema()` + `next_update(timestep, states)`).
`v2ecoli/library/vivarium_bridge.py` lets you run such a process on the
process-bigraph runtime **without rewriting its ports by hand**.

This is the automatic piece extracted from Ryan Spangler's vEcoli
[`composite` branch](https://github.com/vivarium-collective/vEcoli/tree/composite)
(`ecoli/library/bigraph_types.py::translate_ports`). The dual-API bridge base
classes from that branch already live here as
`v2ecoli/library/ecoli_step.py` (`EcoliStep` / `EcoliProcess`); this module adds
the generic `ports_schema` → typed-port inference on top of them.

## When to use it

| Situation | Use |
|-----------|-----|
| Dropping in an unmodified vivarium-1.0 process | `wrap_vivarium_process` (this module) |
| You only need the schema translation | `translate_ports` (this module) |
| A *partitioned* process (`calculate_request` / `evolve_state`) | subclass `v2ecoli.steps.partition.PartitionedProcess` instead |
| Writing a brand-new v2ecoli process from scratch | subclass `EcoliStep` / `EcoliProcess` directly with typed `inputs()`/`outputs()` |

The bridge is for *plain* time-driven (or step) vivarium processes. Partitioned
processes have their own request/evolve split and should use
`PartitionedProcess`.

## Quick start

```python
from vivarium.core.process import Process
from v2ecoli.library.vivarium_bridge import wrap_vivarium_process

# 1. An ordinary vivarium-1.0 process — unchanged.
class Counter(Process):
    name = "counter"
    defaults = {"step_size": 2}

    def ports_schema(self):
        return {"count": {"_default": 0, "_updater": "accumulate"}}

    def next_update(self, timestep, states):
        return {"count": self.parameters["step_size"] * timestep}

# 2. Wrap it. The result is an EcoliProcess subclass.
CounterBridge = wrap_vivarium_process(Counter)
```

`CounterBridge` derives its `inputs()` / `outputs()` automatically from
`Counter().ports_schema()` and routes the process-bigraph
`update(state, interval)` call to `Counter.next_update(timestep, states)`.

### Running it in a composite

```python
from process_bigraph import Composite
from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core

core = build_core()
set_current_core(core)                       # so the bridge can find the core
core.register_link("CounterBridge", CounterBridge)

doc = {
    "count": 0,
    "counter": {
        "_type": "process",
        "address": "local:CounterBridge",
        "config": {"step_size": 2},
        "inputs": {"count": ["count"]},       # wire the port to a store path
        "outputs": {"count": ["count"]},
        "interval": 1.0,
    },
}

composite = Composite({"state": doc}, core=core)
composite.run(4)
assert composite.state["count"] == 8.0       # 2 per step × 4 steps
```

A live, runnable version of this is in
[`tests/test_vivarium_bridge.py`](../tests/test_vivarium_bridge.py)
(`test_real_vivarium_process_runs_in_composite`).

## How port translation works

`translate_ports(core, ports)` walks a vivarium `ports_schema()` dict and infers
a v2ecoli typed-port tree:

| vivarium port schema | becomes |
|----------------------|---------|
| `{"_default": 0}` | `{"_type": "integer", "_default": 0}` |
| `{"_default": [0.0, 0.0]}` | `{"_type": "list[float]", "_default": [0.0, 0.0]}` |
| `{"_default": True, "_updater": "set"}` | `{"_type": "overwrite[boolean]", "_default": True}` |
| `{"_default": ()}` | default normalized to `[]` |
| nested `{"a": {...}, "b": {...}}` | nested dict, recursed |
| store named `bulk` / `bulk_total` | `"bulk_array"` (raw numpy default ignored) |
| store in `UNIQUE_TYPES` (e.g. `RNAs`, `active_ribosome`) | the registered `unique_array[...]` type |

Type inference uses the bigraph-schema `core.infer()` + `render()`; the
`_updater: "set"` flag maps to `overwrite[...]` (matching vivarium's `set`
updater / `_divider: set` semantics). Other updaters (`accumulate`, default)
leave the plain inferred type, and updates accumulate via process-bigraph's
`apply`.

## Caveats

- **Bidirectional ports.** vivarium `ports_schema` is read/write; by default the
  bridge declares every port in *both* `inputs()` and `outputs()` (safe
  over-declaration). Pass `output_ports=[...]` to restrict the write surface:

  ```python
  CounterBridge = wrap_vivarium_process(Counter, output_ports=["count"])
  ```

- **`set_current_core(core)`** must run before instantiating a bridged process
  if you don't pass `core=` explicitly — the bridge needs a core to infer types.

- **`as_step=True`** wraps as an `EcoliStep` (runs to convergence within a tick)
  instead of the default time-driven `EcoliProcess`.

- **No vivarium-core dependency required.** The wrapped class is duck-typed — it
  only needs `ports_schema()` and `next_update()` (or `update()`). The example
  above uses the real `vivarium.core.process.Process` to show interop, but a
  plain class with those two methods works identically.

- **Units.** If the v1 process passes `Unum` quantities, convert them at the
  boundary with `v2ecoli/library/unit_bridge.py` — Unum must not leak into
  v2ecoli internals (see AGENTS.md).
