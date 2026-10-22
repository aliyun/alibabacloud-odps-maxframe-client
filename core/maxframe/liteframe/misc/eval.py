# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import binascii
import operator
import sys
import textwrap
import tokenize
from collections import OrderedDict
from functools import reduce
from io import StringIO

from maxframe.liteframe.core import LITEFRAME_TYPE, LiteFrameData
from maxframe.liteframe.expressions import LiteFrameBaseExpr, _is_bool_arrow_dtype, col
from maxframe.liteframe.utils import normalize_col_name

LOCAL_TAG = "_local_var_"
BACKTICK_TAG = "_backtick_var_"


def _tokenize_str(reader):
    token_generator = tokenize.generate_tokens(reader)

    def _iter_backtick_string(gen, line, back_start):
        for _, tokval, start, _, _ in gen:
            if tokval == "`":
                return (
                    BACKTICK_TAG
                    + binascii.b2a_hex(
                        line[back_start[1] + 1 : start[1]].encode()
                    ).decode()
                )
        else:
            raise SyntaxError(f"backtick quote at {back_start} does not match")

    for toknum, tokval, start, _, line in token_generator:
        if toknum == tokenize.OP:
            if tokval == "@":
                tokval = LOCAL_TAG
            if tokval == "&":
                toknum = tokenize.NAME
                tokval = "and"
            elif tokval == "|":
                toknum = tokenize.NAME
                tokval = "or"
        elif tokval == "`":
            yield tokenize.NAME, _iter_backtick_string(token_generator, line, start)
            continue
        yield toknum, tokval


class _ColumnResolver:
    """Resolve bare column names to unbound ``col(name)`` expressions.

    Resolves against the *target* frame (the frame being mutated by
    assignment), so columns added by earlier lines of a multi-line
    expression resolve to ``col(name)``.  Column names are normalized to
    lowercase to match LiteFrame's case-insensitive storage.  Unknown names
    raise ``KeyError`` so the visitor falls through to the environment
    lookup.
    """

    def __init__(self, frame):
        self._frame = frame

    def __getitem__(self, name):
        normalized = normalize_col_name(name)
        if normalized in self._frame.dtypes.index:
            return col(normalized)
        raise KeyError(name)


class CollectionVisitor(ast.NodeVisitor):
    _op_handlers = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        # ``in`` / ``not in``: test membership of the left operand (a column
        # expression) in the right operand (a constant list).  The DataFrame
        # visitor reverses the operands (``y.isin(x)``) which is incorrect for
        # column-on-left usage; here ``isin`` is dispatched onto the column
        # expression so ``A in [1, 2, 3]`` -> ``col("A").isin([1, 2, 3])``.
        ast.In: lambda x, y: x.isin(y),
        ast.NotIn: lambda x, y: ~x.isin(y),
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Invert: operator.invert,
        # ``not`` keyword maps to logical inversion.  pandas' query treats
        # ``not x`` as ``~x`` on boolean operands; mirror that so
        # ``lf.query("not (A > 2)")`` builds the same filter as ``~(A > 2)``.
        ast.Not: operator.invert,
        ast.And: operator.and_,
        ast.Or: operator.or_,
    }

    def __init__(self, resolvers, target, env):
        self.env = env
        self.target = target
        self.resolvers = resolvers

    def _preparse(self, expr):
        reader = StringIO(expr).readline
        return tokenize.untokenize(list(_tokenize_str(reader)))

    def eval(self, expr, rewrite=True):
        if rewrite:
            expr = self._preparse(expr)
        node = ast.fix_missing_locations(ast.parse(expr))
        return self.visit(node)

    def get_named_object(self, obj_name):
        for resolver in self.resolvers:
            try:
                return resolver[obj_name]
            except (IndexError, KeyError):
                continue
        if obj_name in self.env:
            return self.env[obj_name]
        raise KeyError(f"name {obj_name} is not defined")

    def visit(self, node):
        if isinstance(node, LiteFrameBaseExpr):
            return node
        node_name = node.__class__.__name__
        method = "visit_" + node_name
        try:
            visitor = getattr(self, method)
        except AttributeError:
            raise SyntaxError(
                "Query string contains unsupported syntax: {}".format(node_name)
            )
        return visitor(node)

    def visit_Module(self, node):
        if self.target is None and len(node.body) != 1:
            raise SyntaxError("Only a single expression is allowed")
        # Multiline eval (target is not None) is documented as a sequence of
        # assignments.  Reject mixed statements (e.g. ``A\nC = A + B`` would
        # silently discard the first pure expression) before evaluating any
        # line so a later failure cannot leave the target partially mutated.
        if self.target is not None and len(node.body) > 1:
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    raise SyntaxError(
                        "Multiline expressions must be a sequence of "
                        "assignments; got {}".format(type(stmt).__name__)
                    )
        result = None
        for expr in node.body:
            result = self.visit(expr)
        return result

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        try:
            handler = self._op_handlers[type(node.op)]
        except KeyError:
            raise SyntaxError(f"unsupported operator: {type(node.op).__name__}")
        return handler(left, right)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(n) for n in node.args]
        kwargs = OrderedDict([(kw.arg, self.visit(kw.value)) for kw in node.keywords])
        return func(*args, **kwargs)

    def visit_Compare(self, node):
        ops = node.ops
        comps = node.comparators

        if len(comps) == 1:
            binop = ast.BinOp(op=ops[0], left=node.left, right=comps[0])
            return self.visit(binop)

        left = node.left
        values = []
        for op, comp in zip(ops, comps):
            new_node = ast.Compare(comparators=[comp], left=left, ops=[op])
            left = comp
            values.append(new_node)
        return self.visit(ast.BoolOp(op=ast.And(), values=values))

    def visit_BoolOp(self, node):
        def func(lhs, rhs):
            binop = ast.BinOp(op=node.op, left=lhs, right=rhs)
            return self.visit(binop)

        return reduce(func, node.values)

    def visit_UnaryOp(self, node):
        op = self.visit(node.operand)
        try:
            handler = self._op_handlers[type(node.op)]
        except KeyError:
            raise SyntaxError(f"unsupported operator: {type(node.op).__name__}")
        return handler(op)

    def visit_Name(self, node):
        if node.id.startswith(LOCAL_TAG):
            local_name = node.id.replace(LOCAL_TAG, "")
            return self.env[local_name]
        if node.id.startswith(BACKTICK_TAG):
            local_name = binascii.a2b_hex(
                node.id.replace(BACKTICK_TAG, "").encode()
            ).decode()
            return self.get_named_object(local_name)
        return self.get_named_object(node.id)

    def visit_Constant(self, node):
        return node.value

    # Python 3.7 emits Num/Str/Bytes/NameConstant instead of the unified
    # Constant node (3.8+).  Never called on 3.8+ where the parser
    # produces Constant directly.
    def visit_Num(self, node):
        return node.n

    def visit_Str(self, node):
        return node.s

    visit_Bytes = visit_Str

    def visit_NameConstant(self, node):
        return node.value

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    visit_Tuple = visit_List

    def visit_Assign(self, node):
        if self.target is None:
            raise ValueError("Target not specified for assignment")
        # Require exactly one target node that is a bare ``ast.Name``;
        # reject chained assignment (``a = b = 1``), tuple/sequence unpacking
        # (``A, B = ...``), attribute (``obj.x = ...``), and subscript
        # (``df[0] = ...``) targets before evaluating the assigned value so
        # an invalid target cannot leave a half-mutated frame.
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise SyntaxError(
                "Assignment target must be a single column name; "
                "chained, tuple, attribute, and subscript targets are "
                "not supported"
            )

        # Decode the preparse rewriting: backtick-quoted targets (```C C 2```)
        # decode to the original column name; ``@local`` targets are rejected.
        target_id = node.targets[0].id
        if target_id.startswith(LOCAL_TAG):
            raise SyntaxError(
                f"Cannot assign to a @local variable: @{target_id.replace(LOCAL_TAG, '')}"
            )
        if target_id.startswith(BACKTICK_TAG):
            target = binascii.a2b_hex(
                target_id.replace(BACKTICK_TAG, "").encode()
            ).decode()
        else:
            target = target_id
        value = self.visit(node.value)
        self.target[target] = value

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            resolved = self.visit(node.value)
            return getattr(resolved, node.attr)
        raise ValueError("Invalid Attribute context {0}".format(node.ctx.__name__))

    def visit_Subscript(self, node):
        value = self.visit(node.value)
        sub = self.visit(node.slice)
        return value[sub]

    def visit_Slice(self, node):
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None
        return slice(lower, upper, step)


def liteframe_eval(
    expr,
    parser="maxframe",
    engine=None,
    local_dict=None,
    global_dict=None,
    resolvers=(),
    level=0,
    target=None,
    inplace=False,
):
    """Evaluate a Python expression string against a LiteFrame.

    Column names resolve to :func:`~maxframe.liteframe.col` expressions, so
    arithmetic / comparison / boolean operators build native LiteFrame
    expression trees.  A pure expression returns a one-column LiteFrame;
    an assignment (``C = A + B``) mutates the target LiteFrame via
    ``__setitem__`` and returns the (possibly copied) target.

    ``@local`` references caller-frame variables, and backtick-quoted names
    resolve columns that are not valid Python identifiers.
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a string")

    if parser not in (None, "maxframe"):
        raise NotImplementedError("Does not support parser names other than maxframe")
    if engine is not None:
        raise NotImplementedError("Does not support specifying engine names")

    expr = textwrap.dedent(expr)
    if not expr.strip():
        # Match pandas: a blank (empty or whitespace-only) expression is
        # rejected outright.  Without this, an empty module body makes the
        # visitor return None, which the inplace path treats as a successful
        # assignment and the non-inplace path returns as a silent copy.
        raise ValueError("expr cannot be an empty string")

    try:
        frame = sys._getframe(level + 1)
        # Build the environment so caller-supplied dicts shadow the caller
        # frame, and local_dict shadows global_dict — matching pandas'
        # ``@var`` resolution (locals win over globals on a name clash).
        # Copy frame.f_locals / f_globals into fresh dicts so caller-owned
        # mappings (including the supplied local_dict / global_dict objects)
        # are never mutated.
        env = dict()
        env.update(frame.f_globals)
        if global_dict is not None:
            env.update(global_dict)
        env.update(frame.f_locals)
        if local_dict is not None:
            env.update(local_dict)
    finally:
        del frame

    # ``abs(col)`` reaches ``LiteFrameExpr.__abs__`` via the builtin; seed it
    # unless the caller's environment already shadows it.
    env.setdefault("abs", abs)

    if target is not None and not inplace:
        target = target.copy()

    # The first LiteFrame resolver is the binding source for pure-expression
    # results; resolve column names against the frame being mutated (the
    # target) so columns created by earlier assignment lines of a multi-line
    # expression resolve to ``col(name)``.  For query (target=None) resolve
    # against the source LiteFrame.
    source = next((r for r in resolvers if isinstance(r, LITEFRAME_TYPE)), None)
    resolve_frame = target if target is not None else source
    wrapped_resolvers = []
    if resolve_frame is not None:
        wrapped_resolvers.append(_ColumnResolver(resolve_frame))
    wrapped_resolvers.extend(r for r in resolvers if not isinstance(r, LITEFRAME_TYPE))

    visitor = CollectionVisitor(tuple(wrapped_resolvers), target, env)
    result = visitor.eval(expr)

    if result is None:
        # Last statement was an assignment (or empty): return the target.
        return None if inplace else target

    if isinstance(result, LiteFrameBaseExpr):
        if source is None:
            raise ValueError("No LiteFrame resolver to bind the expression against")
        return source.select(result)

    return result


def lf_eval(lf, expr, inplace=False, **kwargs):
    """Evaluate a string describing operations on LiteFrame columns.

    A pure expression returns a one-column LiteFrame (the LiteFrame analogue
    of a pandas Series).  Assignment expressions (``C = A + B``) add or
    replace a column; ``inplace=True`` mutates *lf*, otherwise a copy is
    returned.

    Supported operators: ``+ - * / // % **``, ``== != < <= > >=``,
    ``& | ~``, unary ``+ -``, ``abs()``, and ``in`` / ``not in`` against a
    constant list.  ``@local`` references caller variables and backtick
    quoting handles column names that are not valid identifiers.

    Parameters
    ----------
    expr : str
        Expression string to evaluate.  Column names resolve to
        :func:`~maxframe.liteframe.col` expressions.
    inplace : bool, default False
        If ``True``, mutate *lf* in place for assignment expressions and
        return ``None``; a pure expression with ``inplace=True`` raises
        ``ValueError``.  If ``False``, return the resulting LiteFrame (for
        a pure expression) or a mutated copy (for an assignment).
    **kwargs
        Additional keyword arguments forwarded to :func:`liteframe_eval`,
        such as ``local_dict``, ``global_dict``, and ``level``.

    Returns
    -------
    LiteFrame or None
        A one-column LiteFrame for a pure expression, a mutated copy for a
        non-inplace assignment, or ``None`` for an inplace assignment.

    See Also
    --------
    query : Query the columns of a LiteFrame with a boolean expression.
    assign : Add new columns to LiteFrame.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})

    A pure expression returns a one-column LiteFrame:

    >>> result = lf.eval('a + b')
    >>> result.execute()
       _expr_0
    0       11
    1       22
    2       33

    An assignment expression adds a column:

    >>> result = lf.eval('c = a * 2')
    >>> result.execute()
       a   b   c
    0  1  10   2
    1  2  20   4
    2  3  30   6
    """
    if inplace and isinstance(lf, LiteFrameData):
        raise NotImplementedError(
            "inplace eval is not supported on a LiteFrameData; call eval on "
            "the owning LiteFrame instead"
        )
    level = kwargs.pop("level", None) or 0
    kwargs["inplace"] = inplace
    val = liteframe_eval(expr, resolvers=(lf,), target=lf, level=level + 1, **kwargs)
    if not inplace:
        return val
    # inplace=True with a pure expression (no assignment) is a silent no-op:
    # the input frame is unchanged yet the result is discarded.  Match the
    # pandas contract and raise so the caller learns the expression mutated
    # nothing.  An inplace assignment returns None from liteframe_eval, so a
    # non-None value here means the expression was pure.
    if val is not None:
        raise ValueError("Cannot operate inplace if there is no assignment")


def lf_query(lf, expr, inplace=False, **kwargs):
    """Query the columns of a LiteFrame with a boolean expression.

    The boolean expression is evaluated with :func:`liteframe_eval` and the
    resulting single-column boolean LiteFrame is used to filter *lf* via
    ``__getitem__``, reusing the existing filter fusion.  ``inplace=True``
    replaces *lf*'s data with the filtered result.

    Parameters
    ----------
    expr : str
        Boolean expression string.  Must evaluate to a single boolean
        column; column names resolve to :func:`~maxframe.liteframe.col`
        expressions.
    inplace : bool, default False
        If ``True``, replace *lf*'s underlying data with the filtered
        result and return ``None``.  If ``False``, return a new filtered
        LiteFrame.
    **kwargs
        Additional keyword arguments forwarded to :func:`liteframe_eval`,
        such as ``local_dict``, ``global_dict``, and ``level``.

    Returns
    -------
    LiteFrame or None
        Filtered LiteFrame when ``inplace=False``; ``None`` when
        ``inplace=True`` (the mutation is applied in place).

    See Also
    --------
    eval : Evaluate a string expression on LiteFrame columns.

    Examples
    --------
    >>> import maxframe.liteframe as mlf
    >>> lf = mlf.LiteFrame({'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]})

    Filter rows where ``a`` is greater than 2:

    >>> result = lf.query('a > 2')
    >>> result.execute()
       a   b
    2  3  30
    3  4  40
    4  5  50

    Compound expressions are supported:

    >>> result = lf.query('(a > 2) & (b < 50)')
    >>> result.execute()
       a   b
    2  3  30
    3  4  40
    """
    if inplace and isinstance(lf, LiteFrameData):
        raise NotImplementedError(
            "inplace query is not supported on a LiteFrameData; call query on "
            "the owning LiteFrame instead"
        )
    level = kwargs.pop("level", None) or 0
    predicate = liteframe_eval(
        expr, resolvers=(lf,), target=None, level=level + 1, **kwargs
    )
    # query must filter on a single boolean column.  A scalar (e.g. ``True``)
    # or a non-boolean column (e.g. an int column passed directly) would
    # leak a KeyError from __getitem__ or silently select the wrong rows.
    if (
        not isinstance(predicate, LITEFRAME_TYPE)
        or predicate.ndim != 2
        or len(predicate.dtypes) != 1
        or not _is_bool_arrow_dtype(predicate.dtypes.iloc[0])
    ):
        raise ValueError("query expression must evaluate to a single boolean column")
    result = lf[predicate]

    if inplace:
        # Swap the underlying data on the owning LiteFrame.  LiteFrameData
        # is rejected above — it is not directly mutable; callers mutate via
        # the LiteFrame entity.
        lf._data.detach(lf)
        lf._data = result.data
        result.data.attach(lf)
    else:
        return result
