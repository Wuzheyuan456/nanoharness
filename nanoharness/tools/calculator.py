"""
安全的数学表达式求值工具 / Safe math expression evaluator tool.

用 AST 白名单解析，只允许算术运算和 math 模块函数，没有任何 exec/eval 安全漏洞 /
Uses AST whitelist parsing — only arithmetic and math module functions, no exec/eval vulnerabilities.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from nanoharness.core.tool_executor import ToolResult, ToolResultStatus

CALCULATOR_DEF = {
    "name": "calculator",
    "description": (
        "对数学表达式求值。支持基本运算（+ - * / ** //）和 math 函数（sqrt, log, sin, cos, floor, ceil 等）。"
        "示例：'2**10'、'sqrt(144)'、'sin(pi/2)'、'log(100, 10)'"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
}

_SAFE_NAMES: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "exp": math.exp,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "math": math,
}

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"不允许的标识符: {node.id}")
    if isinstance(node, ast.Attribute):
        # 允许 math.xxx / allow math.xxx
        obj = _eval_node(node.value)
        val = getattr(obj, node.attr, None)
        if val is None:
            raise ValueError(f"不允许的属性: {node.attr}")
        return val
    if isinstance(node, ast.BinOp):
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
        return op_fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
        return op_fn(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        if not callable(func):
            raise ValueError(f"不可调用: {ast.dump(node.func)}")
        call_args = [_eval_node(a) for a in node.args]
        return func(*call_args)
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


def calculator_fn(tool_input: dict[str, Any], _ctx: Any) -> ToolResult:
    expr = (tool_input.get("expression") or "").strip()
    if not expr:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content="表达式为空，请提供要计算的数学表达式",
            error_code="empty_expression",
        )
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        # 整数结果不显示小数点 / show integer results without decimal
        if isinstance(result, float) and result == int(result) and not math.isinf(result):
            formatted = str(int(result))
        elif isinstance(result, float):
            formatted = f"{result:.10g}"
        else:
            formatted = str(result)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            content=f"{expr} = {formatted}",
            next_action_hint="计算完成，结果已在上方。",
        )
    except Exception as exc:
        return ToolResult(
            status=ToolResultStatus.FAILURE,
            content=f"计算失败: {exc}",
            error_code="calc_error",
            next_action_hint="请检查表达式语法，只支持算术运算和 math 函数。",
        )
