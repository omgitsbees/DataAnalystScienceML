import nltk
import spacy
import tkinter as tk
from tkinter import scrolledtext
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK data 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Tokenization
def tokenize(text):
    return nltk.word_tokenize(text)

# Part-of-speech tagging
def pos_tagging(text):
    tokens = tokenize(text)
    return nltk.pos_tag(tokens)

# Named Entity Recognition
def named_entity_recognition(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Sentiment Analysis
def sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

# NLP Analyzer Class
class NLPAnalyzer:
    def __init__(self, text):
        self.text = text

    def tokenize(self):
        return tokenize(self.text)
    
    def pos_tag(self):
        return pos_tagging(self.text)
    
    def ner(self):
        return named_entity_recognition(self.text)
    
    def sentiment(self):
        return sentiment_analysis(self.text)

# GUI Application
class NLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NLP Analyzer")
        
        self.text_input = tk.Text(root, height=10, width=50)
        self.text_input.pack()

        self.analyze_button = tk.Button(root, text="Analyze", command=self.analyze_text)
        self.analyze_button.pack()

        self.result_text = scrolledtext.ScrolledText(root, height=20, width=70)
        self.result_text.pack()

    def analyze_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if text:
            analyzer = NLPAnalyzer(text)
            tokenization_result = analyzer.tokenize()
            pos_tagging_result = analyzer.pos.tag()
            ner_result = analyzer.ner()
            sentiment_result = analyzer.sentiment()

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Tokenization:\n" + str(tokenization_result) + "\n\n")
            self.result_text.insert(tk.END, "Part-of-Speech Tagging:\n" + str(pos_tagging_result) + "\n\n")
            self.result_text.insert(tk.END, "Named Entity Recognition:\n" + str(ner_result) + "\n\n")
            self.result_text.insert(tk.END, "Sentiment Analysis:\n" + str(sentiment_result) + "\n\n")

# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    app = NLPApp(root)
    root.mainloop()