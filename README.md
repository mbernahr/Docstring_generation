# Automated docstring generation for Python <img src="Icon/Python-Docstring-Generator-Icon.png" alt="Icon" width="80">

This project develops an end-to-end system for the automatic generation and improvement of docstrings in Python files
using Large Language Models (LLMs). The aim is to add missing docstrings to existing scripts as well as to optimize the
content of existing docstrings.

## Repository structure

The repository is organized as follows:

- [Data](Data):  
  Contains the raw data (Python scripts), filtered scripts, and training and test data for fine-tuning.
- [Code](Code):  
  Contains the following scripts and notebooks for dataset processing, analysis, and fine-tuning:
    - `create_finetune_dataset.py`:  
      Creates the dataset used for fine-tuning the model.
    - `docstring_analysis.py`:  
      Analyzes docstrings in Python scripts. This script is used during dataset creation and is also utilized in the
      app.
    - `docstring_generator_app.py`:  
      A complete application that wraps the fine-tuned model, analyzes Python files, and generates docstrings for
      functions.
    - `finetune_LLM_docstring.ipynb`:  
      A Jupyter notebook for fine-tuning the model using LoRA (Low-Rank Adaptation).

- [Results](Results):  
  Contains test and evaluation files, including:
    - `inference_finetuned_LLM_docstring.ipynb`:  
      A Jupyter notebook for evaluating the fine-tuned model's performance. It uses BLEU and ROUGE scores to measure
      similarity with reference docstrings and provides visual comparisons between generated and reference docstrings.

- [Icon](Icon):  
  Contains the application icon that is displayed in the dock when the program is executed.

- [Models](Models):  
  Contains the final checkpoint of the fine-tuned model, which is used in the application for docstring generation.


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

  ```[Function] def function(a, b):\n ... \nreturn list [Docstring] Do X and return a list. [EOS]```

## Fine-Tuning Large Language Model

In this project, the model [CodeLlama](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf)
from [Hugging Face](https://huggingface.co) is fine-tuned using LoRA (Low-Rank Adaptation) on an
extracted and filtered dataset.

1. **Dataset Preparation**:
    - The training and test datasets are loaded from text files containing Python functions and corresponding
      docstrings.
    - The dataset is tokenized to prepare it for fine-tuning.

2. **Model Setup**:
    - The pre-trained model (`meta-llama/CodeLlama-7b-Python-hf`) is loaded from Hugging Face.
    - LoRA is applied to fine-tune the model with efficient parameter adaptation.

3. **Fine-Tuning Configuration**:
    - The training process is configured with parameters such as learning rate, batch size, and gradient accumulation.
    - Additional tools like `wandb` and `mlflow` are used for experiment tracking and monitoring.

4. **Evaluation**:
    - The fine-tuned model is tested using a pipeline on several prompts, and the generated docstrings are evaluated for
      syntax, semantic clarity, and completeness.
    - Evaluation metrics include BLEU and ROUGE scores.

5. **Inference**:
    - The fine-tuned model is saved and used in the application to generate docstrings for Python functions.

## Docstring generation Script

The script requires **Python >= 3.10** and uses the libraries [pandas](https://pandas.pydata.org/) (*BSD 3-Clause
License*), [matplotlib](https://matplotlib.org/) (*Matplotlib License*), and the standard Python libraries `json`, `os`.

### Installation

Clone the repository and install the [required dependencies](Code/requirements.txt) as follows:

```bash
git clone https://github.com/mbernahr/Docstring_generation.git
cd Docstring_generation/Code
```

#### Create virtual environment

- Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

- Windows:

```bash
python -m venv venv
venv\Scripts\activate 
```

#### Install the required dependencies:

```bash
pip install -r ..\requirements.txt
```

### Usage

Before the first use, create an account at https://huggingface.co and create a token. Next, login with the token from huggingface:
```bash
huggingface-cli login
```


Run the script as follows:

```bash
python docstring_generator_app.py
```

### What the Script Does

1. Provides a graphical user interface (GUI) for selecting a Python script to analyze.
2. Extracts all functions from the selected script and displays them in a table, including their current docstring
   quality scores.
3. Allows the user to:
    - Select all functions.
    - Focus on functions with low-quality scores for targeted improvements.
    - Select specific functions manually for docstring generation.
4. Generates new docstrings for missing or inadequate ones using the fine-tuned model.
5. Updates the Python script with the generated docstrings and saves the modified file with a new name in the same
   directory.

The updated file is saved with the suffix `_updated.py`.

### Contributors

- Marius Bernahrndt
- Maximilian Karhausen
