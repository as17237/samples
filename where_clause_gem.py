import sqlparse
from zss import simple_distance, Node

def sql_to_ast(sql_query):
    """
    Converts a SQL WHERE clause to an Abstract Syntax Tree (AST).
    """
    parsed = sqlparse.parse(sql_query)[0]
    where_clause = next((token for token in parsed.tokens if token.ttype is sqlparse.tokens.Where), None)
    
    if where_clause is None:
        return None

    def _recurse(node):
        if isinstance(node, sqlparse.sql.TokenList):
            return Node(str(node.ttype), [_recurse(child) for child in node.tokens])
        elif isinstance(node, sqlparse.sql.Token) and node.ttype is not sqlparse.tokens.Whitespace:
            return Node(str(node.value))
        return None

    return _recurse(where_clause)

def calculate_similarity(sql1, sql2):
    """
    Calculates the similarity score between two SQL WHERE clauses using tree edit distance.
    """
    ast1 = sql_to_ast(sql1)
    ast2 = sql_to_ast(sql2)

    if ast1 is None or ast2 is None:
        return 0.0  # Handle cases where no WHERE clause is found

    distance = simple_distance(ast1, ast2)
    max_distance = max(ast1.get_size(), ast2.get_size())  # Normalize by the larger tree size
    similarity = 1 - (distance / max_distance)
    return similarity

if __name__ == "__main__":
    where_clause1 = input("Enter the first where clause: ")
    where_clause2 = input("Enter the second where clause: ")

    similarity_score = calculate_similarity(where_clause1, where_clause2)

    print(f"Similarity score: {similarity_score:.2f}")
