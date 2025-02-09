#!/usr/bin/env python3
"""
Enhanced SQL Server WHERE Clause Parser with Subtree Similarity Detection

This script parses SQL Server WHERE clauses and builds an AST. It then:
  - Extracts subtrees (of a configurable minimum depth) to identify duplicate branches.
  - Uses canonical representations to detect exact duplicate subtrees.
  - Uses the Zhang-Shasha tree edit distance (via the zss library) to measure near-duplicate similarity.
  
Installation requirements:
  pip install pyparsing zss
"""

import pyparsing as pp
from pyparsing import (
    Word, alphas, alphanums, nums, Combine, QuotedString, Suppress,
    Optional, Group, Forward, delimitedList, oneOf, CaselessKeyword, opAssoc, infixNotation
)

# Enable packrat parsing for performance
pp.ParserElement.enablePackrat()

# -----------------------------
# 1. Define the Grammar
# -----------------------------
identifier = Word(alphas, alphanums + "_$").setName("identifier")
integer = Word(nums)
real = Combine(Word(nums) + '.' + Word(nums))
number = (real | integer).setName("number")
quoted_string = QuotedString("'", escChar='\\').setName("quoted_string")

lparen = Suppress("(")
rparen = Suppress(")")
expr = Forward()

# Function call (e.g. UPPER(country) or ISNULL(market_cap, 0))
function_call = Group(
    identifier("func_name") +
    lparen +
    Optional(delimitedList(expr))("args") +
    rparen
).setName("function_call")

# Atom: a function call, number, quoted string, identifier or a parenthesized expression
atom = (function_call | number | quoted_string | identifier | Group(lparen + expr + rparen)).setName("atom")
expr <<= atom

# SQL operators and keywords
comparison_op = oneOf("< <= > >= = <>", asKeyword=True)
like_op = CaselessKeyword("LIKE")
in_op = CaselessKeyword("IN")
is_op = CaselessKeyword("IS")
null_kw = CaselessKeyword("NULL")
not_kw = CaselessKeyword("NOT")

# Predicates: comparisons, LIKE, IN, IS (NOT) NULL
comparison = Group(expr("left") + comparison_op("op") + expr("right")).setName("comparison")
like_condition = Group(expr("left") + like_op("op") + expr("right")).setName("like_condition")
in_condition = Group(expr("left") + in_op("op") + lparen + delimitedList(expr)("list") + rparen).setName("in_condition")
is_null_condition = Group(expr("left") + is_op("op") + Optional(not_kw("not")) + null_kw("null")).setName("is_null_condition")

predicate = (comparison | like_condition | in_condition | is_null_condition | expr).setName("predicate")

# Boolean expression with proper precedence (NOT > AND > OR)
bool_expr = infixNotation(predicate,
    [
        (CaselessKeyword("NOT"), 1, opAssoc.RIGHT),
        (CaselessKeyword("AND"), 2, opAssoc.LEFT),
        (CaselessKeyword("OR"), 2, opAssoc.LEFT),
    ]
).setName("bool_expr")

# -----------------------------
# 2. AST Node Classes
# -----------------------------
class Node:
    def canonical(self):
        raise NotImplementedError("Subclasses must implement canonical()")

class ValueNode(Node):
    def __init__(self, value):
        self.value = value
    def canonical(self):
        return str(self.value)
    def __repr__(self):
        return self.canonical()

class FunctionNode(Node):
    def __init__(self, func_name, args):
        self.func_name = func_name
        self.args = args  # list of Node objects
    def canonical(self):
        args_can = ", ".join(arg.canonical() for arg in self.args)
        return f"{self.func_name.upper()}({args_can})"
    def __repr__(self):
        return self.canonical()

class ComparisonNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def canonical(self):
        return f"{self.left.canonical()} {self.op} {self.right.canonical()}"
    def __repr__(self):
        return self.canonical()

class LikeNode(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
    def canonical(self):
        return f"{self.left.canonical()} LIKE {self.right.canonical()}"
    def __repr__(self):
        return self.canonical()

class InNode(Node):
    def __init__(self, left, values):
        self.left = left
        self.values = values  # list of Node objects
    def canonical(self):
        # Sorting values makes the ordering irrelevant.
        values_sorted = sorted(v.canonical() for v in self.values)
        return f"{self.left.canonical()} IN ({', '.join(values_sorted)})"
    def __repr__(self):
        return self.canonical()

class IsNullNode(Node):
    def __init__(self, left, is_not=False):
        self.left = left
        self.is_not = is_not
    def canonical(self):
        return f"{self.left.canonical()} IS {'NOT ' if self.is_not else ''}NULL"
    def __repr__(self):
        return self.canonical()

class NotNode(Node):
    def __init__(self, operand):
        self.operand = operand
    def canonical(self):
        return f"NOT {self.operand.canonical()}"
    def __repr__(self):
        return self.canonical()

class AndNode(Node):
    def __init__(self, operands):
        self.operands = operands  # list of Node objects
    def canonical(self):
        sorted_ops = sorted(op.canonical() for op in self.operands)
        return " AND ".join(sorted_ops)
    def __repr__(self):
        return self.canonical()

class OrNode(Node):
    def __init__(self, operands):
        self.operands = operands  # list of Node objects
    def canonical(self):
        sorted_ops = sorted(op.canonical() for op in self.operands)
        return " OR ".join(sorted_ops)
    def __repr__(self):
        return self.canonical()

# -----------------------------
# 3. Build the AST from Parsed Tokens
# -----------------------------
def parse_to_tree(parsed):
    """
    Recursively convert pyparsing output into our AST of Node objects.
    """
    if isinstance(parsed, (list, pp.ParseResults)) and len(parsed) == 1:
        return parse_to_tree(parsed[0])
    
    if isinstance(parsed, pp.ParseResults) and "func_name" in parsed:
        args = []
        if "args" in parsed and parsed["args"]:
            for arg in parsed["args"]:
                args.append(parse_to_tree(arg))
        return FunctionNode(parsed["func_name"], args)
    
    if isinstance(parsed, pp.ParseResults) and "left" in parsed and "op" in parsed:
        left_node = parse_to_tree(parsed["left"])
        op = parsed["op"]
        if op.upper() in ["<", "<=", ">", ">=", "=", "<>"]:
            right_node = parse_to_tree(parsed["right"])
            return ComparisonNode(left_node, op, right_node)
        elif op.upper() == "LIKE":
            right_node = parse_to_tree(parsed["right"])
            return LikeNode(left_node, op, right_node)
        elif op.upper() == "IN":
            values = [parse_to_tree(item) for item in parsed["list"]]
            return InNode(left_node, values)
        elif op.upper() == "IS":
            is_not = bool(parsed.get("not"))
            return IsNullNode(left_node, is_not)
    
    if isinstance(parsed, (list, pp.ParseResults)):
        # Handle NOT operator: ["NOT", operand]
        if len(parsed) == 2 and isinstance(parsed[0], str) and parsed[0].upper() == "NOT":
            return NotNode(parse_to_tree(parsed[1]))
        # Handle AND/OR operators
        if any(isinstance(token, str) and token.upper() in ["AND", "OR"] for token in parsed):
            op_tokens = [token for token in parsed if isinstance(token, str) and token.upper() in ["AND", "OR"]]
            if op_tokens:
                op = op_tokens[0].upper()
                operands = [parse_to_tree(token) for token in parsed if not (isinstance(token, str) and token.upper() in ["AND", "OR"])]
                if op == "AND":
                    return AndNode(operands)
                elif op == "OR":
                    return OrNode(operands)
        return parse_to_tree(parsed[0])
    
    if isinstance(parsed, (str, int, float)):
        return ValueNode(parsed)
    
    raise ValueError(f"Unable to parse element: {parsed}")

def canonicalize_where_clause(where_clause):
    """
    Parse a SQL WHERE clause and return its canonical representation along with the AST.
    
    Returns:
        (canonical_string, ast_root)
    """
    parsed = bool_expr.parseString(where_clause, parseAll=True)
    tree = parse_to_tree(parsed)
    return tree.canonical(), tree

# -----------------------------
# 4. Subtree Similarity Functions
# -----------------------------
def collect_subtrees(node, depth_threshold=1, current_depth=0):
    """
    Recursively collect subtrees (as (node, canonical, depth)) that meet a minimum depth.
    """
    subtrees = []
    if current_depth >= depth_threshold:
        subtrees.append((node, node.canonical(), current_depth))
    # Recurse into children based on node type.
    if isinstance(node, FunctionNode):
        for arg in node.args:
            subtrees.extend(collect_subtrees(arg, depth_threshold, current_depth + 1))
    elif isinstance(node, ComparisonNode):
        subtrees.extend(collect_subtrees(node.left, depth_threshold, current_depth + 1))
        subtrees.extend(collect_subtrees(node.right, depth_threshold, current_depth + 1))
    elif isinstance(node, LikeNode):
        subtrees.extend(collect_subtrees(node.left, depth_threshold, current_depth + 1))
        subtrees.extend(collect_subtrees(node.right, depth_threshold, current_depth + 1))
    elif isinstance(node, InNode):
        subtrees.extend(collect_subtrees(node.left, depth_threshold, current_depth + 1))
        for v in node.values:
            subtrees.extend(collect_subtrees(v, depth_threshold, current_depth + 1))
    elif isinstance(node, IsNullNode):
        subtrees.extend(collect_subtrees(node.left, depth_threshold, current_depth + 1))
    elif isinstance(node, NotNode):
        subtrees.extend(collect_subtrees(node.operand, depth_threshold, current_depth + 1))
    elif isinstance(node, (AndNode, OrNode)):
        for child in node.operands:
            subtrees.extend(collect_subtrees(child, depth_threshold, current_depth + 1))
    return subtrees

def find_duplicate_subtrees(root, depth_threshold=1):
    """
    Identify exact duplicate subtrees (based on canonical representation) that appear
    at or beyond the specified depth.
    
    Returns:
        Dict mapping canonical representation -> list of (node, depth) tuples.
    """
    subtrees = collect_subtrees(root, depth_threshold)
    groups = {}
    for node, canon, depth in subtrees:
        groups.setdefault(canon, []).append((node, depth))
    # Only return groups with more than one occurrence.
    return {canon: nodes for canon, nodes in groups.items() if len(nodes) > 1}

# -----------------------------
# 5. Near-Duplicate Similarity via Tree Edit Distance
# -----------------------------
# Install zss via: pip install zss
import zss

def get_children(node):
    """
    Retrieve children of an AST node for tree edit distance computation.
    """
    if isinstance(node, FunctionNode):
        return node.args
    elif isinstance(node, ComparisonNode):
        return [node.left, node.right]
    elif isinstance(node, LikeNode):
        return [node.left, node.right]
    elif isinstance(node, InNode):
        return [node.left] + node.values
    elif isinstance(node, IsNullNode):
        return [node.left]
    elif isinstance(node, NotNode):
        return [node.operand]
    elif isinstance(node, (AndNode, OrNode)):
        return node.operands
    return []

def node_label(node):
    """
    Use the class name plus canonical representation as the node label.
    """
    return node.__class__.__name__ + ":" + node.canonical()

def insert_cost(node):
    return 1

def remove_cost(node):
    return 1

def update_cost(a, b):
    return 0 if node_label(a) == node_label(b) else 1

def tree_distance(node1, node2):
    """
    Compute the tree edit distance between two AST nodes using the Zhang-Shasha algorithm.
    A distance of 0 indicates an exact match.
    """
    return zss.simple_distance(
        node1, node2,
        get_children=get_children,
        insert_cost=insert_cost,
        remove_cost=remove_cost,
        update_cost=update_cost
    )

# -----------------------------
# 6. Example Usage
# -----------------------------
if __name__ == "__main__":
    clause = """
        (market_cap < 500 AND country = 'UK' AND UPPER(sector) = 'TECHNOLOGY')
        OR (market_cap < 500 AND country = 'UK')
        OR (investment_date IS NOT NULL AND market_cap IN (100, 200, 300))
    """
    canon, tree = canonicalize_where_clause(clause)
    print("Canonical WHERE Clause:")
    print(canon)
    print("\n---\n")
    
    # Identify exact duplicate subtrees at or beyond depth 2.
    duplicates = find_duplicate_subtrees(tree, depth_threshold=2)
    if duplicates:
        print("Exact duplicate subtrees found:")
        for canon_repr, nodes in duplicates.items():
            print(f"Subtree: {canon_repr}")
            for node, depth in nodes:
                print(f"  Depth: {depth}, Node: {node}")
    else:
        print("No exact duplicate subtrees found.")
    
    # Example: Compare two subtrees (if available) using tree edit distance.
    dup_subtrees = list(duplicates.values())
    if dup_subtrees and len(dup_subtrees[0]) >= 2:
        node1 = dup_subtrees[0][0][0]
        node2 = dup_subtrees[0][1][0]
        distance = tree_distance(node1, node2)
        print("\nTree edit distance between two duplicate subtrees:")
        print(distance, "(0 means exact match)")
    else:
        print("\nNot enough duplicate subtrees available for tree distance comparison.")
