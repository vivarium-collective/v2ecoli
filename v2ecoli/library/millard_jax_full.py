"""Full-Millard JAX/Diffrax port via generic SBML AST translation.

Loads an SBML file (default: Millard 2017 BIOMD MODEL1505110000 with 77 species,
68 reactions, 61 function definitions, 3 assignment rules), walks every kinetic
law's MathML AST, inlines function-definition calls, substitutes assignment rules,
and emits a single Python source string that closes over the parameter values and
implements ``dy/dt`` as a pure JAX function.  The string is ``exec``'d once, then
JIT-compiled and integrated with diffrax (Kvaerno5, tight tols) by callers.

Scope: handles AST_PLUS, AST_MINUS, AST_TIMES, AST_DIVIDE, AST_FUNCTION_POWER,
AST_FUNCTION_LN, AST_FUNCTION (user-defined, recursively inlined), AST_NAME,
AST_REAL.  Anything else raises; the Millard 2017 SBML uses only those.

Used by ``scripts/run_jax_millard_full.py``; not imported by composites/steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import libsbml
import numpy as np


# ---------------------------------------------------------------------------
# AST → Python source translation
# ---------------------------------------------------------------------------

_BINOPS = {
    libsbml.AST_PLUS: "+",
    libsbml.AST_MINUS: "-",
    libsbml.AST_TIMES: "*",
    libsbml.AST_DIVIDE: "/",
}


def _ast_to_py(node, name_map: dict[str, str], func_bodies: dict, depth: int = 0) -> str:
    """Recursive AST → Python expression string.

    name_map: maps SBML names (species, params, compartments) to the Python
    variable names visible at runtime (e.g. ``y[3]``, ``p_PGI_Keq``, ``1.0``).
    func_bodies: maps function-definition id → (formal_args, body_ast).
    """
    t = node.getType()

    if t == libsbml.AST_REAL or t == libsbml.AST_INTEGER or t == libsbml.AST_RATIONAL:
        return repr(float(node.getValue()))

    if t == libsbml.AST_NAME:
        nm = node.getName()
        if nm in name_map:
            return name_map[nm]
        raise KeyError(f"unmapped name '{nm}' (depth={depth})")

    if t in _BINOPS:
        op = _BINOPS[t]
        # libsbml AST is n-ary for + and *, binary for - and /
        nc = node.getNumChildren()
        if nc == 0:
            # zero-arity '+' = 0, zero-arity '*' = 1 (libsbml quirks)
            return "0.0" if t == libsbml.AST_PLUS else "1.0"
        if nc == 1 and t == libsbml.AST_MINUS:
            return f"(-({_ast_to_py(node.getChild(0), name_map, func_bodies, depth + 1)}))"
        parts = [_ast_to_py(node.getChild(i), name_map, func_bodies, depth + 1) for i in range(nc)]
        return "(" + f" {op} ".join(parts) + ")"

    if t == libsbml.AST_FUNCTION_POWER or t == libsbml.AST_POWER:
        base = _ast_to_py(node.getChild(0), name_map, func_bodies, depth + 1)
        exp = _ast_to_py(node.getChild(1), name_map, func_bodies, depth + 1)
        return f"({base} ** {exp})"

    if t == libsbml.AST_FUNCTION_LN:
        arg = _ast_to_py(node.getChild(0), name_map, func_bodies, depth + 1)
        return f"jnp.log({arg})"

    if t == libsbml.AST_FUNCTION_LOG:
        arg = _ast_to_py(node.getChild(node.getNumChildren() - 1), name_map, func_bodies, depth + 1)
        return f"(jnp.log({arg}) / jnp.log(10.0))"

    if t == libsbml.AST_FUNCTION_EXP:
        arg = _ast_to_py(node.getChild(0), name_map, func_bodies, depth + 1)
        return f"jnp.exp({arg})"

    if t == libsbml.AST_FUNCTION:
        # user-defined function -- inline by substituting formal args with the
        # passed (already-translated) expressions in a fresh name_map
        fname = node.getName()
        if fname not in func_bodies:
            raise KeyError(f"unknown function '{fname}'")
        formals, body = func_bodies[fname]
        if node.getNumChildren() != len(formals):
            raise ValueError(
                f"function {fname} expects {len(formals)} args, got {node.getNumChildren()}"
            )
        # translate each actual-arg expression in CALLER scope first
        actual_exprs = [
            _ast_to_py(node.getChild(i), name_map, func_bodies, depth + 1)
            for i in range(node.getNumChildren())
        ]
        # then translate the body with formals -> wrapped actual-expr substrings
        inner_map = {formal: f"({expr})" for formal, expr in zip(formals, actual_exprs)}
        return _ast_to_py(body, inner_map, func_bodies, depth + 1)

    raise NotImplementedError(
        f"AST type {t} not implemented (depth={depth}, node={libsbml.formulaToL3String(node)})"
    )


# ---------------------------------------------------------------------------
# Build a JAX RHS from an SBML model
# ---------------------------------------------------------------------------


@dataclass
class JaxModel:
    sbml_path: str
    species_ids: list[str] = field(default_factory=list)
    state_species_ids: list[str] = field(default_factory=list)  # non-bc, non-const
    y0: np.ndarray | None = None
    rhs: Callable | None = None
    source: str = ""
    n_reactions: int = 0
    # populated for debugging
    fixed_species: dict[str, float] = field(default_factory=dict)
    assignment_targets: list[str] = field(default_factory=list)


def build_jax_model(sbml_path: str) -> JaxModel:
    """Parse the SBML file and return a JaxModel with a JIT-able rhs."""
    import jax  # noqa: F401 (used in exec'd source)
    import jax.numpy as jnp  # noqa: F401 (used in exec'd source)

    reader = libsbml.SBMLReader()
    doc = reader.readSBML(sbml_path)
    if doc.getNumErrors() > 0:
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                raise RuntimeError(f"SBML parse error: {err.getMessage()}")
    m = doc.getModel()

    # 1. compartments
    compartments = {m.getCompartment(i).getId(): float(m.getCompartment(i).getSize())
                    for i in range(m.getNumCompartments())}

    # 2. species: separate dynamic (state) from constant/boundary
    species_ids = [m.getSpecies(i).getId() for i in range(m.getNumSpecies())]
    state_species: list[str] = []
    fixed: dict[str, float] = {}
    init_conc: dict[str, float] = {}
    for i in range(m.getNumSpecies()):
        s = m.getSpecies(i)
        if s.isSetInitialConcentration():
            init_conc[s.getId()] = float(s.getInitialConcentration())
        elif s.isSetInitialAmount():
            comp_size = compartments.get(s.getCompartment(), 1.0)
            init_conc[s.getId()] = float(s.getInitialAmount()) / comp_size
        else:
            init_conc[s.getId()] = 0.0
        if s.getBoundaryCondition() or s.getConstant():
            fixed[s.getId()] = init_conc[s.getId()]
        else:
            state_species.append(s.getId())

    # 3. global parameters
    global_params = {m.getParameter(i).getId(): float(m.getParameter(i).getValue())
                     for i in range(m.getNumParameters())}

    # 4. function definitions: id -> (formal_args, body_ast)
    func_bodies = {}
    for i in range(m.getNumFunctionDefinitions()):
        fd = m.getFunctionDefinition(i)
        formals = [fd.getArgument(j).getName() for j in range(fd.getNumArguments())]
        func_bodies[fd.getId()] = (formals, fd.getBody())

    # 5. assignment rules: variable -> body_ast. We will substitute the rule
    #    expression in-place when the variable name appears in a kinetic law.
    assignment_asts: dict[str, libsbml.ASTNode] = {}
    for i in range(m.getNumRules()):
        rule = m.getRule(i)
        if rule.getElementName() == "assignmentRule":
            assignment_asts[rule.getVariable()] = rule.getMath()
        else:
            raise NotImplementedError(f"rule type {rule.getElementName()} not supported")

    # Build the state-index map: state species -> y[i]
    state_index = {sid: i for i, sid in enumerate(state_species)}

    # ------------------------------------------------------------------
    # Construct name_map used to translate every kinetic-law AST.
    # ------------------------------------------------------------------
    base_name_map: dict[str, str] = {}
    # state species
    for sid, idx in state_index.items():
        base_name_map[sid] = f"y[{idx}]"
    # fixed (boundary/constant) species -- emit numeric literals
    for sid, val in fixed.items():
        base_name_map[sid] = repr(val)
    # also species that aren't in state nor fixed (shouldn't happen, but
    # defensively map via init_conc)
    for sid in species_ids:
        if sid not in base_name_map:
            base_name_map[sid] = repr(init_conc.get(sid, 0.0))
    # compartments
    for cid, size in compartments.items():
        base_name_map[cid] = repr(size)
    # global parameters
    for pid, val in global_params.items():
        base_name_map[pid] = repr(val)

    # Assignment rules: when the variable name appears inside a kinetic law,
    # we replace it with the inlined translated expression.  Build a helper
    # name_map that maps each assignment variable to a *placeholder name*,
    # and we'll do a final-stage textual substitution.  Simpler approach:
    # translate each assignment-rule body once into a Python expression, and
    # use it as the mapping for the variable.
    assignment_exprs: dict[str, str] = {}
    for var, ast in assignment_asts.items():
        # When translating the rule body, the variable itself shouldn't appear;
        # use base_name_map (other species/params).
        assignment_exprs[var] = "(" + _ast_to_py(ast, base_name_map, func_bodies) + ")"
    # Merge into name_map so subsequent kinetic-law translations substitute.
    name_map = dict(base_name_map)
    name_map.update(assignment_exprs)

    # ------------------------------------------------------------------
    # Translate each kinetic law into a Python expression v_i.
    # Build the stoichiometric matrix as a constant numpy array.
    # ------------------------------------------------------------------
    n_rxn = m.getNumReactions()
    n_state = len(state_species)
    S = np.zeros((n_state, n_rxn))  # rows: state species, cols: reactions
    rate_exprs: list[str] = []

    for r_idx in range(n_rxn):
        rx = m.getReaction(r_idx)
        kl = rx.getKineticLaw()
        # local parameters go into a per-reaction overlay name_map
        local_map = dict(name_map)
        for k in range(kl.getNumLocalParameters()):
            lp = kl.getLocalParameter(k)
            local_map[lp.getId()] = repr(float(lp.getValue()))
        for k in range(kl.getNumParameters()):
            lp = kl.getParameter(k)
            local_map[lp.getId()] = repr(float(lp.getValue()))
        try:
            expr = _ast_to_py(kl.getMath(), local_map, func_bodies)
        except Exception as e:
            raise RuntimeError(
                f"failed to translate kinetic law for reaction {rx.getId()}: {e}\n"
                f"formula: {libsbml.formulaToL3String(kl.getMath())}"
            ) from e
        rate_exprs.append(expr)

        # Fill stoichiometry for state species only
        for j in range(rx.getNumReactants()):
            sref = rx.getReactant(j)
            sid = sref.getSpecies()
            if sid in state_index:
                S[state_index[sid], r_idx] -= float(sref.getStoichiometry())
        for j in range(rx.getNumProducts()):
            sref = rx.getProduct(j)
            sid = sref.getSpecies()
            if sid in state_index:
                S[state_index[sid], r_idx] += float(sref.getStoichiometry())

    # Account for compartment volumes: the kinetic-law value is *extent of
    # reaction per unit time* (rate × compartment, since formulas usually
    # include the `cell *` prefix).  dC/dt = (1/V) * (sum_j s_ij * v_j).
    # Each state species' compartment volume:
    state_comp_vol = np.array([
        compartments[m.getSpecies(species_ids.index(sid)).getCompartment()]
        for sid in state_species
    ])

    # ------------------------------------------------------------------
    # Emit Python source for rhs(t, y, args) -> dy/dt.
    # ------------------------------------------------------------------
    src_lines = [
        "def _rhs(t, y, args):",
        "    # state unpacking implicit via y[i]",
    ]
    for i, expr in enumerate(rate_exprs):
        src_lines.append(f"    v{i} = {expr}")
    # Build dy/dt
    src_lines.append("    return jnp.array([")
    for i, sid in enumerate(state_species):
        terms = []
        for j in range(n_rxn):
            coef = float(S[i, j])
            if coef == 0:
                continue
            if coef == 1:
                terms.append(f"v{j}")
            elif coef == -1:
                terms.append(f"-v{j}")
            else:
                terms.append(f"({coef!r} * v{j})")
        if not terms:
            inner = "0.0"
        else:
            inner = " + ".join(terms).replace("+ -", "- ")
        vol = float(state_comp_vol[i])
        if vol == 1.0:
            src_lines.append(f"        ({inner}),  # d{sid}/dt")
        else:
            src_lines.append(f"        ({inner}) / {vol!r},  # d{sid}/dt")
    src_lines.append("    ])")

    source = "\n".join(src_lines)

    # Compile.
    ns: dict = {}
    import jax.numpy as _jnp
    ns["jnp"] = _jnp
    exec(compile(source, f"<millard_jax:{sbml_path}>", "exec"), ns)
    rhs = ns["_rhs"]

    y0 = np.array([init_conc[sid] for sid in state_species], dtype=np.float64)

    return JaxModel(
        sbml_path=sbml_path,
        species_ids=species_ids,
        state_species_ids=state_species,
        y0=y0,
        rhs=rhs,
        source=source,
        n_reactions=n_rxn,
        fixed_species=fixed,
        assignment_targets=list(assignment_exprs.keys()),
    )
