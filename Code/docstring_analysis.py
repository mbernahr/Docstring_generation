import os
import ast
import subprocess
import tempfile
import re


def run_pylint(filename):
    """Execute pylint to check for missing docstrings."""
    result = subprocess.run(
        [
            'pylint',
            '--disable=all',
            '--enable=missing-docstring',
            filename
        ],
        capture_output=True,
        text=True
    )
    return result.stdout


def run_pydocstyle(filename):
    """Execute pydocstyle to check for PEP 257 compliance."""
    result = subprocess.run(
        [
            'pydocstyle',
            '--add-ignore=D100',
            filename,
        ],
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip().split('\n')


def check_docstring_content(function_docstring):
    """
    Check the content of a function's docstring for specific sections.

    Identifies if the docstring likely contains Args, Returns, Raises, or Examples.
    Only meaningful if docstring has multiple lines.

    Args:
        function_docstring (str): The docstring of the function to check.

    Returns:
        dict: A dictionary indicating the presence of the following sections:
              - 'HasDocstring' (bool): True if a docstring is present at all.
              - 'Args' (bool): True if 'Args:', 'Arguments:', 'Params:', or 'Parameters:' is present.
              - 'Returns' (bool): True if 'Returns:' or 'Yields:' is present.
              - 'Raises' (bool): True if 'Raises:' or 'Exceptions:' is present.
              - 'Examples' (bool): True if 'Examples:', 'Example:', or 'Usage:' is present.
              If the docstring is None, all keys will return False.
    """
    if not function_docstring:
        return {
            'HasDocstring': False,
            'HasMultipleLines': False,
            'Args': False,
            'Returns': False,
            'Raises': False,
            'Examples': False
        }

    has_multiple_line = len(function_docstring.strip().split("\n")) <= 1

    args_keywords = ['Args:', 'Arguments:', 'Params:', 'Parameters:']
    returns_keywords = ['Returns:', 'Yields:']
    raises_keywords = ['Raises:', 'Exceptions:']
    examples_keywords = ['Examples:', 'Example:', 'Usage:']

    results = {
        'HasDocstring': True,
        'HasMultipleLines': has_multiple_line,
        'Args': any(keyword in function_docstring for keyword in args_keywords),
        'Returns': any(keyword in function_docstring for keyword in returns_keywords),
        'Raises': any(keyword in function_docstring for keyword in raises_keywords),
        'Examples': any(keyword in function_docstring for keyword in examples_keywords),
    }
    return results


def is_complex_function(num_args, code_length):
    """
    Determine if a function is considered 'complex'.

    Heuristic:
    - Complex if more than 1 argument OR more than 20 lines of code.

    Args:
        num_args (int): Number of arguments in the function.
        code_length (int): Number of lines in the function source.

    Returns:
         bool: True if the function has more than 1 argument or is longer than 20 lines.
    """
    has_many_args = num_args > 1
    is_long = code_length > 20

    return has_many_args or is_long


def calculate_docstring_score(pylint_output, pydocstyle_output, docstring_content, is_complex, num_args):
    """
    Calculate a heuristic score based on pylint, pydocstyle output and docstring content.

    Basis:
    - Start from pylint rating * 10 (0-100 scale).
    - Apply discounts based on pydocstyle errors.
    - If docstring is multi-line and the function is complex,
      - Check 'Args' only if num_args > 1.
      - Always check 'Returns', 'Raises', 'Examples'.

    Args:
        pylint_output (str): The raw pylint output.
        pydocstyle_output (list): Lines returned by pydocstyle.
        docstring_content (dict): Content checks from check_docstring_content().
        is_complex (bool): Whether the function is considered complex.
        num_args (int): Number of arguments the function has.

    Returns:
        float: A score between 0 and 100 representing docstring quality.
    """
    match = re.search(r"Your code has been rated at ([\d.]+)/10", pylint_output)
    if match:
        score = int(float(match.group(1)) * 10)
    else:
        score = 50

    discounts = {
        "D100": 5,  # Missing module docstring
        "D101": 8,  # Missing class docstring
        "D102": 30,  # Missing function docstring
        "D103": 5,  # Missing public function docstring
        "D201": 5,  # No blank lines allowed before function docstring
        "D202": 5,  # No blank lines allowed after function docstring
        "D204": 3,  # Missing blank line after docstring
        "D205": 5,  # 1 Blank line required between summary line and description
        "D210": 2,  # Whitespace around docstring text
        "D300": 10,  # Use tripple double quotes
        "D400": 3,  # Missing period at the end of the docstring
        "D401": 5,  # First line should be in imperative mood
        "D403": 2,  # First word not capitalized
        "D419": 5,  # Empty docstring
        # Special discounts for docstring_content
        "HasMultipleLines": 0,
        "Args": 8,
        "Returns": 8,
        "Raises": 2,
        "Examples": 2
    }

    for line in pydocstyle_output:
        if ": " in line:
            parts = line.split(": ")
            if len(parts) > 1:
                error_code = parts[0].strip()
                if error_code in discounts:
                    score -= discounts[error_code]
                else:
                    score -= 1

    if not docstring_content['HasDocstring']:
        score -= 30
    else:
        if docstring_content.get('HasMultipleLines') and is_complex:
            if num_args > 1 and not docstring_content['Args']:
                score -= discounts['Args']
            if not docstring_content['Returns']:
                score -= discounts['Returns']
            if not docstring_content['Raises']:
                score -= discounts['Raises']
            if not docstring_content['Examples']:
                score -= discounts['Examples']

    return score


def remove_docstring_from_ast(func_node):
    """Extract the Docstring from an ast.FunctionDef node."""
    if (func_node.body and
            isinstance(func_node.body[0], ast.Expr) and
            (
                    (
                            isinstance(func_node.body[0].value, ast.Str)
                    )
                    or
                    (
                            isinstance(func_node.body[0].value, ast.Constant) and
                            isinstance(func_node.body[0].value.value, str)
                    )
            )
    ):
        func_node.body.pop(0)
    return func_node


def extract_functions_from_file(filename):
    """
    Extract all functions from a Python file.

    Returns:
        list of tuples: (function_name, function_source, function_docstring, num_args, code_length)
    """
    with open(filename, "r", encoding="utf-8") as file:
        source = file.read()
        tree = ast.parse(source, filename=filename)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            num_args = len(node.args.args)

            start_line = node.lineno - 1
            end_line = node.end_lineno
            code_length = end_line - start_line

            function_docstring = ast.get_docstring(node)
            remove_docstring_from_ast(node)
            function_source = ast.unparse(node)

            functions.append((function_name, function_source, function_docstring, num_args, code_length))

    return functions


def analyze_function(function_name, function_source, function_docstring, num_args, code_length):
    """
    Analyze a single function with pylint, pydocstyle, and heuristic checks.

    Returns:
        dict: Analysis results including pylint output, pydocstyle output, docstring content, final score.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(function_source)
        temp_filename = temp_file.name

    pylint_output = run_pylint(temp_filename)
    pydocstyle_output = run_pydocstyle(temp_filename)
    docstring_content = check_docstring_content(function_docstring)
    os.unlink(temp_filename)

    complex_fn = is_complex_function(num_args, code_length)
    score = calculate_docstring_score(pylint_output, pydocstyle_output, docstring_content, complex_fn, num_args)

    result = {
        "name": function_name,
        "score": score,
        "pylint": pylint_output,
        "pydocstyle": pydocstyle_output,
        "docstring_content": docstring_content,
        "is_complex": complex_fn,
        "num_args": num_args,
        "code_length": code_length
    }

    return result


def analyze_file(filename):
    """
    Analyze a Python file for docstring quality.

    - Extracts all functions.
    - For each function, runs analysis and scores docstring quality.
    - Prints out results.

    Args:
        filename (str): Path to the Python file to analyze.

    Returns:
        list: A list of dicts containing the analysis for each function.
    """
    print(f'Analyzing file: {filename}')

    functions = extract_functions_from_file(filename)
    if not functions:
        print('No functions found in the file.')
        return []

    results = []

    for func in functions:
        function_name, function_source, function_doc, num_args, code_length = func
        print(f'\n--- Analyzing function: {function_name} ---')
        analysis_results = analyze_function(function_name, function_source, function_doc, num_args, code_length)
        start_line = func[3]
        results.append({
            "Name": function_name,
            "Docstring": function_doc,
            "Source": function_source,
            "IsComplex": analysis_results["is_complex"],
            "Score": analysis_results["score"],
            "StartLine": start_line,
            "CodeLength": code_length
        })

    return results
