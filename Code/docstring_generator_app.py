import ast
import os
import platform
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import torch
from docstring_analysis import analyze_file

# Set transformers logging verbosity to error
logging.set_verbosity_error()


class DocstringApp(TkinterDnD.Tk):  # Inherit from TkinterDnD.Tk for drag-and-drop
    def __init__(self):
        super().__init__()
        self.title("Docstring Generator")

        # Variables
        self.filename = None
        self.functions = []
        self.text_generator = None

        # UI Setup
        self.setup_ui()
        self.setup_drag_and_drop()
        self.setup_icon()

        # Load model in the background
        threading.Thread(target=self.init_model, daemon=True).start()

    def setup_ui(self):
        """Setup the user interface components."""
        self.setup_frames()
        self.setup_buttons()
        self.setup_progress_bar()

    def setup_icon(self):
        """Setup the application icon."""
        icon_path = "../Icon/Python-Docstring-Generator-Icon.png"
        system = platform.system()
        try:
            if system == "Windows":
                self.iconbitmap(icon_path)
            else:
                icon = Image.open(icon_path)
                icon_photo = ImageTk.PhotoImage(icon)
                self.wm_iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Error loading icon: {e}")

    def setup_frames(self):
        """Setup frames and labels for the UI."""
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

    def setup_buttons(self):
        """Setup buttons for UI interactions."""
        # Buttons for docstring generation
        self.generate_all_button = tk.Button(
            self, text="Generate docstrings for all functions",
            command=lambda: self.generate_docstrings(for_all=True), state=tk.DISABLED
        )
        self.generate_all_button.pack(pady=5)

        self.generate_low_button = tk.Button(
            self, text="Generate docstrings for functions with a low score",
            command=lambda: self.generate_docstrings(for_all=False), state=tk.DISABLED
        )
        self.generate_low_button.pack(pady=5)

    def setup_progress_bar(self):
        """Setup the progress bar and status indicators."""
        # Progress indicator for model loading
        self.status_frame = tk.Frame(self)
        self.status_frame.pack(pady=5, padx=10, anchor="ne")

        self.status_light = tk.Canvas(self.status_frame, width=20, height=20, highlightthickness=0, bg=self.cget("bg"))
        self.status_circle = self.status_light.create_oval(2, 2, 18, 18, fill="red")
        self.status_light.pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(self.status_frame, text="Loading checkpoint shards...", fg="red")
        self.status_label.pack(side=tk.RIGHT)

        # Progress bar for docstring generation
        self.progress_frame = tk.Frame(self)
        self.progress_frame.pack(pady=5, padx=10)
        self.progress_label = tk.Label(self.progress_frame, text="Progress: 0%", fg="black")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=5)

    def setup_drag_and_drop(self):
        """Configure drag-and-drop functionality."""
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_file_drop)

    def on_file_drop(self, event):
        file_path = event.data.strip('{}')
        self.load_file(file_path)

    def load_file(self, file_path=None):
        """Load a Python file and analyze its functions."""
        if not file_path:
            file_path = filedialog.askopenfilename(filetypes=[("Python Files", "*.py")])

        if not self.is_valid_python_file(file_path):
            messagebox.showerror("Error", "Please select a valid Python file.")
            return

        self.filename = file_path
        self.file_label.config(text=f"Selected file: {os.path.basename(self.filename)}", fg="black")
        self.analyze_file()

    def is_valid_python_file(self, file_path):
        """Check if the given file path points to a valid Python file."""
        return file_path and file_path.endswith(".py")

    def analyze_file(self):
        """Analyze the selected Python file for functions."""
        self.tree.delete(*self.tree.get_children())
        self.functions = analyze_file(self.filename)

        if not self.functions:
            messagebox.showinfo("No functions", "The selected file does not contain any functions.")
            return

        for func in self.functions:
            status = "\U0001F7E2" if func["Score"] >= 75 else "\U0001F534"
            self.tree.insert("", "end", values=(func["Name"], func["Score"], status))

        is_model_ready = self.text_generator is not None
        self.generate_all_button.config(state=tk.NORMAL if is_model_ready else tk.DISABLED)
        self.generate_low_button.config(state=tk.NORMAL if is_model_ready else tk.DISABLED)

    def init_model(self):
        """Initialize the language model."""
        try:
            model_path = "../Models/meta-llama/CodeLlama-7b-Python-hf_run5e7bfd2ad529/checkpoint-200"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using {device} for model loading.")

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=None).to(
                device)
            model.eval()

            self.text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                           device=0 if device == "cuda" else -1)
            self.update_loading_status(100, ready=True)
        except Exception as e:
            messagebox.showerror("Error", f"Model could not be loaded: {e}")
            self.update_loading_status(0, ready=False)

    def update_loading_status(self, percent, ready=False):
        """Update the status indicator for the model."""
        if ready:
            self.status_label.config(text="Model ready", fg="green")
            self.status_light.itemconfig(self.status_circle, fill="green")
        else:
            self.status_label.config(text="Loading checkpoint shards...", fg="red")
            self.status_light.itemconfig(self.status_circle, fill="red")
        self.update_idletasks()

    def generate_docstrings(self, for_all=True):
        """Generate docstrings for all or only for functions with a low score."""
        target_functions = self.get_target_functions(for_all)
        if not target_functions:
            messagebox.showinfo("No functions", "There are no functions with a low score.")
            return

        # Set progress to startvalue
        self.progress_bar["value"] = 1
        self.progress_label.config(text="Progress: 1%")
        self.update_idletasks()

        # Start the generation process in a separate thread
        threading.Thread(
            target=self._generate_docstrings_thread, args=(target_functions,), daemon=True
        ).start()

    def _generate_docstrings_thread(self, target_functions):
        """Threaded function to generate docstrings."""
        file_content = self.read_file(self.filename)
        updated_content = self.process_functions(file_content, target_functions)
        updated_filename = self.filename.replace(".py", "_updated.py")
        self.save_updated_file(updated_filename, updated_content)

        messagebox.showinfo("Finish", f"All docstrings have been generated. Updated file saved as {updated_filename}.")
        self.reset_progress()

    def get_target_functions(self, for_all):
        """Return the target functions for docstring generation."""
        return self.functions if for_all else [f for f in self.functions if f["Score"] < 75]

    def process_functions(self, file_content, target_functions):
        """Generate docstrings and update file content."""
        total_functions = len(target_functions)

        for i, func in enumerate(target_functions, 1):
            docstring = self.generate_docstring_for_function(func)
            file_content = self.insert_docstring_into_file(file_content, func['Name'], docstring)
            self.update_progress(i, total_functions)

        return file_content

    def generate_docstring_for_function(self, func):
        """Generate a docstring for a given function."""
        prompt = f"[Function]\n{func['Source']}\n[Docstring]\n"
        print(f"\n--- Generating Docstring: {func['Source']} ---")

        with torch.no_grad():
            response = self.text_generator(prompt, max_new_tokens=250, do_sample=False)[0]['generated_text']
        processed_response = self.postprocess_response(response)
        docstring_start = processed_response.find("[Docstring]") + len("[Docstring]")
        docstring_end = processed_response.find("[EOS]")
        cleaned_response = processed_response[docstring_start:docstring_end].strip()

        print(f"\n--- Generated Docstring: {func['Name']} \n {cleaned_response} ---")
        return cleaned_response

    def postprocess_response(self, response):
        """Postprocess the response from the model."""
        eos_marker = "[EOS]"
        if eos_marker in response:
            return response.split(eos_marker)[0] + eos_marker
        return "[WARNING: NO EOS MARKER]" + response

    def insert_docstring_into_file(self, file_content, func_name, docstring):
        """Insert the generated docstring into the file."""
        tree = ast.parse("".join(file_content))
        new_lines = file_content[:]

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start_line = node.lineno - 1

                # Remove existing docstring if present
                if (
                        node.body and
                        isinstance(node.body[0], ast.Expr) and
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)
                ):
                    end_line = node.body[0].end_lineno - 1
                    del new_lines[start_line + 1: end_line + 1]

                indentation = " " * (len(file_content[start_line]) - len(file_content[start_line].lstrip()))
                if '\n' not in docstring.strip():
                    new_lines.insert(start_line + 1, f'{indentation}    """{docstring.strip()}"""\n')
                else:
                    docstring_lines = [f'{indentation}    """'] + \
                                      [f'{indentation}    {line.strip()}' for line in docstring.strip().splitlines()] + \
                                      [f'{indentation}    """']
                    new_lines[start_line + 1:start_line + 1] = [line + '\n' for line in docstring_lines]
        return new_lines

    def read_file(self, filename):
        """Read the content of a file."""
        with open(filename, "r") as file:
            return file.readlines()

    def save_updated_file(self, filename, file_content):
        """Save the updated file content."""
        with open(filename, "w") as file:
            file.writelines(file_content)

    def update_progress(self, current, total):
        """Update the progress bar."""
        progress_percent = max(1, int((current/total)*100))
        self.progress_bar["value"] = progress_percent
        self.progress_label.config(text=f"Progress: {progress_percent}%")
        self.update_idletasks()

    def reset_progress(self):
        """Reset the progress bar."""
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Progress: 0%")


if __name__ == "__main__":
    app = DocstringApp()
    app.mainloop()
