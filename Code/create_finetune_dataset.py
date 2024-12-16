import os
import random
from docstring_analysis import analyze_file


def analyze_directory(directory_path):
    """
    Analyze all Python files in a given directory recursively.

    Args:
        directory_path (str): The path to the directory to analyze.

    Returns:
        dict: A dictionary where keys are file paths, and values are lists of results
              for each function in the corresponding file.
    """
    results = {}

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Analyzing file: {file_path}")
                try:
                    results[file_path] = analyze_file(file_path)
                except Exception as e:
                    print(f'Error analyzing {file_path}: {e}')

    return results


def save_results_to_txt(results, output_file, threshold=70):
    """
    Save analyzed results to a .txt file in the specified format.

    Args:
        results (dict): The analysis results from analyze_directory.
        output_file (str): The path to the output .txt file.
        threshold (int): The minimum score a function must have to be saved in the output file.

    Returns:
        int: The total number of lines written to the output file.
    """
    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as file:
        for file_path, functions in results.items():
            for function in functions:
                if function['Docstring'] and function['Score'] > threshold:
                    function_code = function['Source'].replace("\n", "\\n")

                    function_docstring = function['Docstring']
                    function_docstring = function_docstring.replace("\n", "\\n")

                    file.write(f'[Function] {function_code} [Docstring] {function_docstring}\n')
                    total_lines += 1

    print(f'Results saved to {output_file}')
    return total_lines


def train_test_split(input_file, train_file, test_file, train_ratio=0.7):
    """
    Split the content of a file into training and test data files.

    Args:
        input_file (str): The path to the input file.
        train_file (str): The path to the output training file.
        test_file (str): The path to the output test file.
        train_ratio (float): The ratio of data to be used for training (default 0.7).

    Returns:
        None: Write the training and test data to separate files.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_point = int(len(lines) * train_ratio)

    train_data = lines[:split_point]
    test_data = lines[split_point:]

    with open(train_file, "w", encoding="utf-8") as train_f:
        train_f.writelines(train_data)

    with open(test_file, "w", encoding="utf-8") as test_f:
        test_f.writelines(test_data)

    print(f"Training data written to {train_file}")
    print(f"Test data written to {test_file}")


def main():
    preselected_files = '../../MLRepositories-master/Preselected_files'
    finetune_data = '../Data/finetune_data.txt'
    train_data = '../Data/docstring_training_data.txt'
    test_data = '../Data/docstring_test_data.txt'

    results = analyze_directory(preselected_files)
    total_lines = save_results_to_txt(results, finetune_data)
    train_test_split(finetune_data, train_data, test_data)

    print(f"\nCreating dataset with {total_lines} lines complete!")


if __name__ == '__main__':
    main()
