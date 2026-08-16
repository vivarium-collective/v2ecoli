"""Shared regression-test helper for item 57: composite construction must
never leave a Link's pre-built ``instance`` with an unbound ``.core``.

Real defect, confirmed live and reproduced locally: some code paths embed
an already-constructed Step/Process instance into a document (rather than
declaring it by address+config for process-bigraph's own realize path to
build) without threading ``core=`` through. ``.core`` is only ever read
later, by ``bigraph_schema``'s ``Link.serialize``
(``instance.core.access(...)``), so this ships a composite that builds and
runs without error and only crashes at the very last step of a real run —
confirmed live via sim 154 (sms-ecoli build 64, commit c2ae8eb) and via a
local repro against real ``ecoli_baseline``/``baseline_millard`` composites.
"""


def unbound_core_instances(state):
    """Every ``(path, class_name)`` under ``state`` whose Link-shaped node
    has an ``instance`` with ``.core is None``. Empty means every embedded
    instance in this composite's real, constructed state is safely bound."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            instance = node.get("instance")
            if instance is not None and hasattr(instance, "core") and instance.core is None:
                found.append((path, type(instance).__name__))
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, (list, tuple)):
            for idx, v in enumerate(node):
                walk(v, path + (idx,))

    walk(state, ())
    return found
