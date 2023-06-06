import ast
import re
from typing import List, Dict
import inspect

def analyze_code(code: str) -> List[str]:
    """
    Analyzes the provided code for potential errors or inefficiencies.
    Returns a list of messages describing any issues found.
    """
    messages = []

    # check for unused variables
    tree = ast.parse(code)
    used_vars = set()

    class VariableVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

    VariableVisitor().visit(tree)

    all_vars = {node.id for node in tree.body if isinstance(node, ast.Assign) for target in node.targets for node in
                ast.walk(target) if isinstance(node, ast.Name)}

    unused_vars = all_vars - used_vars
    if unused_vars:
        messages.append(f"Warning: unused variable(s) found: {', '.join(sorted(unused_vars))}")

    # check for potential performance issues
    regex = r"(for|while).*range\((\d+)\):"
    match = re.search(regex, code)
    if match:
        loop_type = match.group(1)
        loop_range = int(match.group(2))
        if loop_type == "for" and loop_range > 1000:
            messages.append("Warning: potential performance issue with for loop using large range")
        elif loop_type == "while" and loop_range > 100000:
            messages.append("Warning: potential performance issue with while loop using very large range")

    # check for division by zero
    if "/ 0" in code or "/0" in code:
        messages.append("Warning: division by zero found")

    return messages


def check_syntax(code: str) -> bool:
    """
    Checks if the provided code has any syntax errors.
    Returns True if the code is valid, False otherwise.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def find_functions(code: str) -> List[str]:
    """
    Finds and returns the names of all functions defined in the provided code.
    """
    tree = ast.parse(code)
    function_names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    return function_names


def find_imports(code: str) -> List[str]:
    """
    Finds and returns the names of all modules imported in the provided code.
    """
    tree = ast.parse(code)
    import_names = [node.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    return import_names


def count_lines_of_code(code: str) -> int:
    """
    Counts and returns the number of lines of executable code in the provided code.
    """
    tree = ast.parse(code)
    lines_of_code = sum(1 for node in tree.body if isinstance(node, ast.stmt))
    return lines_of_code


def count_comments(code: str) -> int:
    """
    Counts and returns the number of comment lines in the provided code.
    """
    comment_lines = sum(1 for line in code.split("\n") if line.strip().startswith("#"))
    return comment_lines


def count_blank_lines(code: str) -> int:
    """
    Counts and returns the number of blank lines in the provided code.
    """
    blank_lines = sum(1 for line in code.split("\n") if not line.strip())
    return blank_lines


def get_function_docstring(code: str, function_name: str) -> str:
    """
    Finds and returns the docstring for the specified function in the provided code.
    """
    # parse the code into an AST (Abstract Syntax Tree)
    tree = ast.parse(code)

    # find the function definition node with the specified name
    func_def_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]

    if not func_def_nodes:
        raise ValueError(f"No function definition found for function name '{function_name}'")

    # get the docstring for the function definition node
    docstring = ast.get_docstring(func_def_nodes[0])

    if not docstring:
        raise ValueError(f"No docstring found for function '{function_name}'")

    return docstring


def get_function_arguments(code: str, function_name: str) -> dict:
    """
    Returns a dictionary of argument names and their default values for the specified function in the provided code.
    """
    # parse the code into an AST (Abstract Syntax Tree)
    tree = ast.parse(code)

    # find the function definition node with the specified name
    func_def_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]

    if not func_def_nodes:
        raise ValueError(f"No function definition found for function name '{function_name}'")

    # get the argument names and default values for the function definition node
    arg_defaults = [(arg.arg, ast.literal_eval(arg.value)) for arg in func_def_nodes[0].args.args if arg.annotation is not ast.NameConstant]

    # build a dictionary of argument names and default values
    arg_dict = {arg: default for arg, default in arg_defaults}

    return arg_dict


def get_module_functions(module: str) -> list:
    """
    Returns a list of function names defined in the specified module.
    """
    # import the module
    imported_module = __import__(module)

    # get the names of all the objects in the module
    module_objects = dir(imported_module)

    # filter the module objects to just functions
    module_functions = [o for o in module_objects if inspect.isfunction(getattr(imported_module, o))]

    return module_functions


def get_function_source_code(code: str, function_name: str) -> str:
    """
    Returns the source code for the specified function in the provided code.
    """
    # parse the code into an AST (Abstract Syntax Tree)
    tree = ast.parse(code)

    # find the function definition node with the specified name
    func_def_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]

    if not func_def_nodes:
        raise ValueError(f"No function definition found for function name '{function_name}'")

    # get the source code for the function definition node
    source_code = ast.get_source_segment(code, func_def_nodes[0])

    return source_code


def get_function_return_type(code: str, function_name: str) -> str:
    """
    Finds and returns the return type annotation for the specified function in the provided code.
    """
    # parse the code into an AST (Abstract Syntax Tree)
    tree = ast.parse(code)

    # find the function definition node with the specified name
    func_def_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name]

    if not func_def_nodes:
        raise ValueError(f"No function definition found for function name '{function_name}'")

    # extract the docstring for the function
    func_def_node = func_def_nodes[0]
    docstring_node = ast.get_docstring(func_def_node)
    docstring = docstring_node.strip() if docstring_node else ''

    return docstring

def extract_functions(code: str) -> List[str]:
    """
    Extracts the names of all functions defined in the provided code.
    """
    tree = ast.parse(code)

    # extract the names of all function definition nodes
    func_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]

    return func_names

def extract_imports(code: str) -> List[str]:
    """
    Extracts the names of all imported modules in the provided code.
    """
    tree = ast.parse(code)

    # extract the names of all import nodes
    import_names = []
    for n in tree.body:
        if isinstance(n, ast.Import):
            import_names.extend([a.name for a in n.names])
        elif isinstance(n, ast.ImportFrom):
            import_names.append(n.module)
            import_names.extend([a.name for a in n.names])

    return import_names

def extract_variables(code: str) -> List[str]:
    """
    Extracts the names of all variables defined in the provided code.
    """
    tree = ast.parse(code)

    # extract the names of all variable definition nodes
    var_names = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            var_names.extend([t.id for t in n.targets])

    return var_names

def extract_classes(code: str) -> List[str]:
    """
    Extracts the names of all classes defined in the provided code.
    """
    tree = ast.parse(code)

    # extract the names of all class definition nodes
    class_names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]

    return class_names

def extract_function_calls(code: str) -> List[str]:
    """
    Extracts the names of all function calls in the provided code.
    """
    tree = ast.parse(code)

    # extract the names of all function call nodes
    func_call_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                func_call_names.append(node.func.attr)

    return func_call_names

def extract_variables_from_function(function_code: str) -> List[str]:
    """
    Extracts the names of all variables used within a function from its source code.
    """
    # parse the function's AST
    tree = ast.parse(function_code)

    # extract the names of all variable nodes
    var_names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)]

    return var_names






