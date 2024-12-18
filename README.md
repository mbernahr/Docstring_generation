# Automated docstring generation for Python

This project develops an end-to-end system for the automatic generation and improvement of docstrings in Python files
using Large Language Models (LLMs). The aim is to add missing docstrings to existing scripts as well as to optimize the
content of existing docstrings.

## Repository structure

The repository is organized as follows:

- [Data](Data):  
  Contains the raw data (Python scripts), filtered scripts, and training and test data for fine-tuning.
- [Code](Code):  
  Contains scripts for filtering the dataset, analyzing the docstrings, fine-tuning the LLM and inserting new
  docstrings.
- [Results](Results):  
  Contains output files, e.g. scripts with improved docstrings, log files of the evaluation and an evaluation report.

## Background

In software development, docstrings play a decisive role in the readability and maintainability of code.
However, manual creation is often neglected. With the increasing availability of LLMs, this process can be automated.
This project aims to automatically generate missing docstrings in Python functions and to improve the quality of
existing docstrings.

## Data Basis

- **Source Code Corpus** (~25,000 GitHub repositories, Python scripts):
    - The initial pre-processed training data for fine-tuning was taken
      from [MLRepositories](https://github.com/TechDom/MLRepositories?tab=readme-ov-file#begleitmaterial-für-projekt-8-automated-docstring-generation-for-python-scripts).
    - The provided `filter_sample_scripts.py` script (minimally adapted) was used with adjusted conditions to filter the
      original ~25,000 scripts down to about ~5,000 scripts. The adapted conditions:
      ```python
      # Define conditions for filtering
      conditions = {
          "lines": lambda x: 20 <= x <= 2000,  # between 20 and 2000 lines
          "functions": lambda x: x >= 2,       # at least two functions
          "functions_with_docstring": lambda x: x >= 2, # at least two functions with a docstring
          "average_line_length": lambda x: x <= 120,     # max. 120 characters per line
      }
      ```
    - After filtering, additional scripts from this reduced dataset were analyzed with our code, resulting in a
      fine-tuning dataset of approximately 13,000 functions with docstrings.

  A typical entry in the final dataset looks like this:
  
  ```[Function] def function(a, b):\n ... \nreturn list [Docstring] Do X and return a list.```

## Large Language Model

In this project, a model from [Hugging Face](https://huggingface.co), such as Llama 3, is used and fine-tuned on the
extracted and filtered dataset.

## Docstring generation Script

The script requires **Python >= 3.10** and uses the libraries [pandas](https://pandas.pydata.org/) (*BSD 3-Clause
License*), [matplotlib](https://matplotlib.org/) (*Matplotlib License*), and the standard Python libraries `json`, `os`.

### Installation

Clone the repository and install the [required dependencies](Code/requirements.txt) as follows:

```bash
git clone https://github.com/mbernahr/Docstring_generation.git
cd Docstring_generation/Code
pip install -r requirements.txt
```

### Usage

Run the script as follows:

```bash
python docstring_generation.py
```

### What the Script Does

1. Guides the user to provide a Python script to analyze.
2. Extracts all functions from the input script.
3. Analyzes the quality of existing docstrings.
4. Generates new docstrings for missing or inadequate ones using the fine-tuned LLM.
5. Outputs a new Python file with the updated docstrings.

The updated file will be saved in the Results directory.

### Contributors

- Marius Bernahrndt
- Maximilian Karhausen
