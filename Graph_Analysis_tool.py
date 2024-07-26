import networkx as nx
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import urllib.request
import gzip
import shutil

def create_graph():
    return nx.Graph()

def add_node(graph, node):
    graph.add_node(node)

def add_edge(graph, node1, node2):
    graph.add_edge(node1, node2)

def display_graph(graph):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=15)
    plt.show()

def shortest_path(graph, start_node, end_node):
    try:
        return nx.shortest_path(graph, source=start_node, target=end_node)
    except nx.NetworkXNoPath:
        return None

def is_connected(graph):
    return nx.is_connected(graph)

def centrality_measures(graph):
    degree_centrality = nx.degree_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)
    return {
        "degree_centrality": degree_centrality,
        "closeness_centrality": closeness_centrality,
        "betweenness_centrality": betweenness_centrality
    }

def download_and_extract(url, output_file):
    gz_file = output_file + ".gz"
    urllib.request.urlretrieve(url, gz_file)
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def load_graph_from_csv(file_path):
    graph = create_graph()
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            add_edge(graph, row[0], row[1])
    return graph

def load_graph_ui(file_path, output_text):
    if file_path:
        graph = load_graph_from_csv(file_path)
        display_graph(graph)
        measures = centrality_measures(graph)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, "Graph loaded and displayed from file.\n")
        output_text.insert(tk.END, "Centrality Measures:\n")
        for measure, values in measures.items():
            output_text.insert(tk.END, f"{measure}:\n")
            for node, value in values.items():
                output_text.insert(tk.END, f"  {node}: {value:.4f}\n")
    else:
        messagebox.showerror("Error", "No file selected")

def main():
    root = tk.Tk()
    root.title("Graph Analysis Tool")
    root.geometry("800x600")

    output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20, font=("Arial", 12))
    output_text.pack(pady=10)

    download_button = tk.Button(root, text="Download and Load Graph", command=lambda: download_and_extract(
        "https://snap.stanford.edu/data/facebook_combined.txt.gz", "facebook_combined.txt"))
    download_button.pack(pady=10)

    load_button = tk.Button(root, text="Load Graph from File", command=lambda: load_graph_ui("facebook_combined.txt", output_text))
    load_button.pack(pady=10)

    select_button = tk.Button(root, text="Select CSV File and Load Graph", command=lambda: load_graph_ui(
        filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")]), output_text))
    select_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
