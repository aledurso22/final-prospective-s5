"""AST helpers for structural assertions about source files.

A substring search cannot establish that a symbol, definition, import or
call exists: the searched text occurs in the searching assertion itself, in
docstrings, and in comments. That is not hypothetical -- an assertion reading

    assert "def _stable_model_cells" not in source

could never pass, because the literal it looks for is part of the file it
reads. Every structural question is answered here by parsing instead.

No JAX, no GPU: these run anywhere, which is the point. Structural
regressions should fail on a laptop, not after a cluster round trip.
"""

import ast
import io


def parse(path):
    return ast.parse(io.open(path).read(), filename=path)


def parse_source(source):
    return ast.parse(source)


def function_names(tree):
    """Every function defined anywhere in the tree, sync or async."""
    return {node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def defines_function(tree, name):
    return name in function_names(tree)


def class_names(tree):
    return {node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)}


def assigned_names(tree, module_level_only=True):
    """Names bound by assignment; module level only by default."""
    nodes = tree.body if module_level_only else list(ast.walk(tree))
    names = set()
    def add_target(target):
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                add_target(element)
        elif isinstance(target, ast.Starred):
            add_target(target.value)

    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                add_target(target)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            names.add(node.target.id)
    return names


def imported_names(tree):
    """(module, name) pairs for every import, plus bare module imports."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.name, alias.asname or alias.name))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out.add((node.module or "", alias.name))
    return out


def imports_from(tree, module_suffix):
    """Does the file import this module, or a name of it?

    `from package import certification as CERT` records the module as
    `package` and the name as `certification`, so both sides are checked.
    """
    for module, name in imported_names(tree):
        if module and module.endswith(module_suffix):
            return True
        if name == module_suffix or name.endswith("." + module_suffix):
            return True
    return False


def loop_iterables(node):
    """Dotted/rendered iterables of every `for` loop in the node."""
    out = []
    for child in ast.walk(node):
        if isinstance(child, ast.For):
            dotted = _dotted(child.iter)
            if dotted is None and isinstance(child.iter, ast.Call):
                function = _dotted(child.iter.func)
                arguments = [_dotted(a) or ast.dump(a)
                             for a in child.iter.args]
                dotted = f"{function}({', '.join(arguments)})"
            out.append(dotted or ast.dump(child.iter))
    return out


def _dotted(node):
    """Render an attribute/name chain as a dotted string, or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def called_names(tree):
    """Dotted names of everything CALLED in the tree."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted:
                out.add(dotted)
    return out


def calls(tree, dotted_name):
    """Is `dotted_name` called, allowing an aliased leading component?"""
    if dotted_name in called_names(tree):
        return True
    tail = dotted_name.split(".")[-2:]
    if len(tail) == 2:
        suffix = ".".join(tail)
        return any(name.endswith(suffix) for name in called_names(tree))
    return False


def function_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    return None


def keyword_defaults(node):
    """{argument: literal default} for one function definition."""
    if node is None:
        return {}
    arguments = node.args.args + node.args.kwonlyargs
    defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                + list(node.args.defaults) + list(node.args.kw_defaults))
    out = {}
    for argument, default in zip(arguments, defaults):
        if default is None:
            continue
        try:
            out[argument.arg] = ast.literal_eval(default)
        except (ValueError, TypeError):
            out[argument.arg] = ast.dump(default)
    return out


def has_loop(node):
    return any(isinstance(child, (ast.For, ast.While, ast.AsyncFor))
               for child in ast.walk(node))


def loop_kinds(node):
    return [type(child).__name__ for child in ast.walk(node)
            if isinstance(child, (ast.For, ast.While, ast.AsyncFor))]


def _bound_in(node):
    """Every name a function binds: arguments, assignments, loops, with,
    comprehensions, except-handlers, walruses, nested definitions, imports."""
    bound = set()
    arguments = node.args
    for group in (arguments.posonlyargs, arguments.args, arguments.kwonlyargs):
        bound.update(argument.arg for argument in group)
    for extra in (arguments.vararg, arguments.kwarg):
        if extra:
            bound.add(extra.arg)
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            bound.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                ast.ClassDef)):
            bound.add(child.name)
            if child is not node:
                inner = child.args if hasattr(child, "args") else None
                if inner:
                    for group in (inner.posonlyargs, inner.args,
                                  inner.kwonlyargs):
                        bound.update(argument.arg for argument in group)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.Lambda):
            inner = child.args
            for group in (inner.posonlyargs, inner.args, inner.kwonlyargs):
                bound.update(argument.arg for argument in group)
            for extra in (inner.vararg, inner.kwarg):
                if extra:
                    bound.add(extra.arg)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
        elif isinstance(child, ast.Global) or isinstance(child, ast.Nonlocal):
            bound.update(child.names)
    return bound


def imported_bindings(tree):
    """The names an import actually BINDS, honouring `as` aliases.

    `imported_names` reports (module, imported name) for describing where
    something comes from; for scope questions what matters is the local
    binding, which is the alias when there is one.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
    return bound


def module_scope(tree):
    """Names visible to every function in the module."""
    scope = set(assigned_names(tree, module_level_only=False))
    scope |= function_names(tree) | class_names(tree) | imported_bindings(tree)
    return scope


def undefined_names(path):
    """Names a function reads that nothing in reach ever binds.

    A cheap guard against the commonest edit accident: replacing the code
    that produced a value while leaving a reader of it behind. It has
    already caught `POWER_CEILING` and `gated_errors` after the statements
    that defined them were removed.
    """
    import builtins

    tree = parse(path)
    known = module_scope(tree) | set(dir(builtins)) | {"__name__", "__file__",
                                                       "__doc__"}
    missing = {}

    def visit(node, inherited):
        """A nested function sees its enclosing function's names too."""
        bound = _bound_in(node) | inherited
        loaded = {child.id for child in ast.walk(node)
                  if isinstance(child, ast.Name)
                  and isinstance(child.ctx, ast.Load)}
        unresolved = sorted(loaded - bound)
        if unresolved:
            missing[node.name] = unresolved
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and child is not node:
                visit(child, bound)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit(node, known)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef,
                                       ast.AsyncFunctionDef)):
                    visit(member, known | {"self"})
    return missing
