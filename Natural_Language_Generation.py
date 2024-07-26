import tkinter as tk
from tkinter import scrolledtext
from transformers import pipeline

# Set up the NLG model
generator = pipeline('text-generation', model='gpt2')

class NLGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Natural Language Generation App")

        # Input text label and entry
        self.label = tk.Label(root, text="Enter your prompt:")
        self.label.pack(pady=10)

        self.prompt_entry = tk.Entry(root, width=50)
        self.prompt_entry.pack(pady=10)

        # Generate button
        self.generate_button = tk.Button(root, text="Generate", command=self.generate_text)
        self.generate_button.pack(pady=10)

        # Output text box
        self.output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20)
        self.output_text.pack(pady=10)

    def generate_text(self):
        prompt = self.prompt_entry.get()
        if prompt:
            generated = generator(prompt, max_length=100, num_return_sequences=1)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.INSERT, generated[0]['generated_text'])

if __name__ == "__main__":
    root = tk.Tk()
    app = NLGApp(root)
    root.mainloop()
