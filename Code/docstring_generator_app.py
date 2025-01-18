from transformers import logging
from tkinterdnd2 import DND_FILES, TkinterDnD
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import torch
from docstring_analysis import analyze_file
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import ast

logging.set_verbosity_error()


class DocstringApp(TkinterDnD.Tk):  # Inherit from TkinterDnD.Tk for drag-and-drop
    def __init__(self):
        super().__init__()
        self.title("Docstring Generator")

        # Frame for file selection
        self.frame = tk.Frame(self)
        self.frame.pack(pady=10, padx=10)

        self.label = tk.Label(self.frame, text="Select Python file or drop here:")
        self.label.pack()

        self.file_label = tk.Label(self.frame, text="No file selected", fg="gray")
        self.file_label.pack()

        self.browse_button = tk.Button(self.frame, text="Select file", command=self.load_file)
        self.browse_button.pack(pady=5)

        # Treeview for functional analysis
        self.tree_frame = tk.Frame(self)
        self.tree_frame.pack(pady=10, padx=10)

        self.tree = ttk.Treeview(self.tree_frame, columns=("Function", "Score", "Status"), show="headings", height=10)
        self.tree.heading("Function", text="Function")
        self.tree.heading("Score", text="Score")
        self.tree.heading("Status", text="Status")
        self.tree.column("Function", width=200, anchor="w")
        self.tree.column("Score", width=100, anchor="center")
        self.tree.column("Status", width=100, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Progress indicator for model loading
        self.status_frame = tk.Frame(self)
        self.status_frame.pack(pady=5, padx=10, anchor="ne")

        self.status_light = tk.Canvas(self.status_frame, width=20, height=20, highlightthickness=0, bg=self.cget("bg"))
        self.status_circle = self.status_light.create_oval(2, 2, 18, 18, fill="red")
        self.status_light.pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(self.status_frame, text="Loading checkpoint shards...", fg="red")
        self.status_label.pack(side=tk.RIGHT)

        # Buttons for docstring generation
        self.generate_all_button = tk.Button(self, text="Generate docstrings for all functions",
                                             command=lambda: self.generate_docstrings(for_all=True), state=tk.DISABLED)
        self.generate_all_button.pack(pady=5)

        self.generate_low_button = tk.Button(self, text="Generate docstrings for functions with a low score",
                                             command=lambda: self.generate_docstrings(for_all=False), state=tk.DISABLED)
        self.generate_low_button.pack(pady=5)

        # Progress indicator for docstring generation
        self.progress_frame = tk.Frame(self)
        self.progress_frame.pack(pady=5, padx=10)
        self.progress_label = tk.Label(self.progress_frame, text="Progress: 0%", fg="black")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=5)

        # Variables
        self.filename = None
        self.functions = []
        self.text_generator = None

        # Drag-and-Drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_file_drop)

        # Load model in the background
        threading.Thread(target=self.init_model, daemon=True).start()

    def on_file_drop(self, event):
        file_path = event.data.strip('{}')
        self.load_file(file_path)

    def load_file(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])

        if not file_path or not file_path.endswith(".py"):
            messagebox.showerror("Error", "Please select a valid Python file.")
            return

        self.filename = file_path
        self.file_label.config(text=f"Selected file: {os.path.basename(self.filename)}", fg="black")

        # Functional analysis
        self.analyze_file()

    def analyze_file(self):
        self.tree.delete(*self.tree.get_children())
        self.functions = analyze_file(self.filename)

        if not self.functions:
            messagebox.showinfo("No functions", "The selected file does not contain any functions.")
            return

        for func in self.functions:
            # Determine green and red circle
            status = "\U0001F7E2" if func["Score"] >= 75 else "\U0001F534"

            # Add function, score, and status to tree
            self.tree.insert("", "end", values=(func["Name"], func["Score"], status))

        self.generate_all_button.config(state=tk.NORMAL if self.text_generator else tk.DISABLED)
        self.generate_low_button.config(state=tk.NORMAL if self.text_generator else tk.DISABLED)

    def init_model(self):
        """Initialise the language model."""
        try:
            model_path = "../Models/meta-llama/CodeLlama-7b-Python-hf_run5e7bfd2ad529/checkpoint-200"

            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA is available. Using GPU.")
            else:
                device = "cpu"
                print("No GPU available. Using CPU.")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            print("Loading the model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map=None
            )

            model = model.to(device)
            model.eval()

            self.text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                           device=0 if device == "cuda" else -1)

            # Loading progress completed
            self.update_loading_status(100, ready=True)
            print("Model successfully loaded.")

        except Exception as e:
            messagebox.showerror("Error", f"The model could not be loaded: {e}")
            self.update_loading_status(0, ready=False)

    def update_loading_status(self, percent, ready=False):
        """Update the charge indicator."""
        if ready:
            self.status_label.config(text="Model ready", fg="green")
            self.status_light.itemconfig(self.status_circle, fill="green")
        else:
            self.status_label.config(text=f"Loading checkpoint shards...", fg="red")
            self.status_light.itemconfig(self.status_circle, fill="red")
        self.update_idletasks()

    def generate_docstrings(self, for_all=True):
        """Generate docstrings for all or only for functions with a low score."""
        print(f'\n--- Generate Docstrings ---')
        if not self.functions or not self.text_generator:
            messagebox.showerror("Error", "No functions or no model available.")
            return

        target_functions = self.functions if for_all else [f for f in self.functions if f["Score"] < 75]

        if not target_functions:
            messagebox.showinfo("No functions", "There are no functions with a low score.")
            return

        file_content = self.read_file(self.filename)

        total_functions = len(target_functions)
        self.progress_bar["maximum"] = total_functions
        self.progress_bar["value"] = 0

        for i, func in enumerate(target_functions, 1):
            cleaned_docstring = self.generate_docstring_for_function(func)
            file_content = self.insert_docstring_into_file(file_content, func['Name'], cleaned_docstring)
            self.update_progress(i, total_functions)

        # Save Updated File
        updated_filename = self.filename.replace(".py", "_updated.py")
        self.save_updated_file(updated_filename, file_content)
        messagebox.showinfo("Finish", f"All docstrings have been generated. Updated file saved as {updated_filename}.")

        # Reset progress
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Progress: 0%")

    def postprocess_response(self, response):
        """Remove everything in the response text from the first occurrence of [EOS]."""
        eos_marker = "[EOS]"
        if eos_marker in response:
            return response.split(eos_marker)[0] + eos_marker
        else:
            print('Warning: No [EOS] marker found in response.')
            return "[WARNING: NO EOS MARKER]" + response

    def read_file(self, filename):
        """Read the content of a file and return it as a list of lines."""
        with open(filename, "r") as file:
            return file.readlines()

    def generate_docstring_for_function(self, func):
        """Generate a docstring for a given function."""
        prompt = f"[Function]\n{func['Source']}\n[Docstring]\n"
        print(f"\n--- Generating Docstring: {func['Source']} ---")
        with torch.no_grad():
            response = self.text_generator(prompt, max_new_tokens=250, do_sample=False)[0]['generated_text']

        processed_response = self.postprocess_response(response)
        docstring_start = processed_response.find("[Docstring]") + len("[Docstring]")
        docstring_end = response.find("[EOS]")
        cleaned_response = response[docstring_start:docstring_end].strip()

        print(f"\n--- Generated Docstring: {func['Name']} \n {cleaned_response} ---")
        return cleaned_response

    def insert_docstring_into_file(self, file_content, func_name, docstring):
        """
        Replace or add a docstring for the specified function using AST.

        Args:
            file_content (list): The file content as a list of lines.
            func_name (str): The name of the function to update.
            docstring (str): The new docstring to insert.

        Returns:
            list: Updated file content with the new docstring.
        """
        tree = ast.parse("".join(file_content))
        new_lines = file_content[:]

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start_line = node.lineno - 1
                end_line = node.body[0].lineno - 1 if node.body else start_line + 1

                # Remove the old docstring (if available)
                if (
                        node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)
                ):
                    end_line = node.body[0].end_lineno - 1
                    del new_lines[start_line + 1: end_line + 1]

                indentation = " " * (len(file_content[start_line]) - len(file_content[start_line].lstrip()))

                # Check whether the docstring is single-line or multi-line
                if '\n' not in docstring.strip():
                    # Single-line
                    docstring_line = f'{indentation}    """{docstring.strip()}"""'
                    new_lines.insert(start_line + 1, docstring_line + '\n')
                else:
                    # Multi-line
                    docstring_lines = [f'{indentation}    """']
                    for line in docstring.strip().splitlines():
                        docstring_lines.append(f'{indentation}    {line.strip()}')
                    docstring_lines.append(f'{indentation}    """')
                    new_lines[start_line + 1:start_line + 1] = [line + '\n' for line in docstring_lines]

        return new_lines

    def save_updated_file(self, filename, file_content):
        """Save the updated file content to a new file."""
        with open(filename, "w") as file:
            file.writelines(file_content)

    def update_progress(self, current, total):
        """Update the progress bar and label."""
        self.progress_bar["value"] = current
        self.progress_label.config(text=f"Progress: {int((current / total) * 100)}%")
        self.update_idletasks()


if __name__ == "__main__":
    app = DocstringApp()
    app.mainloop()
