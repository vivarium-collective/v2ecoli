import typing
from plum import dispatch
from dataclasses import dataclass, is_dataclass, field

from bigraph_schema.schema import Node, String, Float
from bigraph_schema.methods import infer, set_default, realize, render, wrap_default


@dataclass(kw_only=True)
class Method(Node):
    module: String = field(default_factory=String)
    instance: object = field(default_factory=object)
    attribute: String = field(default_factory=String)

    def _serialize_state(self, state):
        if isinstance(state, dict):
            return state
        return {
            'module': str(self.module),
            'instance': str(self.instance),
            'attribute': self.attribute,
        }


@infer.dispatch
def infer(core, value: typing.Callable, path: tuple=()):
    if hasattr(value, '__self__'):
        data = {
            'module': value.__module__,
            'instance': value.__self__.__class__.__name__,
            'attribute': value.__func__.__name__}
    else:
        data = {
            'module': value.__module__,
            'instance': None,
            'attribute': value.__name__}

    method = Method(**data)

    return set_default(method, value), []


@realize.dispatch
def realize(core, schema: Method, encode, path=()):
    if callable(encode):
        return schema, encode, []
    elif isinstance(encode, dict):
        import importlib
        module_name = encode.get('module') or str(schema.module)
        instance_name = encode.get('instance') or str(schema.instance)
        attribute_name = encode.get('attribute') or str(schema.attribute)

        mod = importlib.import_module(module_name)

        if instance_name and instance_name != 'None':
            cls = getattr(mod, instance_name)
            func = getattr(cls, attribute_name)
        else:
            func = getattr(mod, attribute_name)

        return schema, func, []
    else:
        return schema, encode, []

@render.dispatch
def render(schema: Method, defaults=False):
    data = {
        '_type': 'method',
        'module': schema.module,
        'instance': str(schema.instance),
        'attribute': schema.attribute}

    return wrap_default(schema, data) if defaults else data
