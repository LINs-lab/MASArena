import sqlite3
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

## ADD YOUR OWN PATHS HERE
DB_FILE = "data/results/melo_memory/smolagents/chroma.sqlite3"
GRAPH_FILE = "data/results/melo_memory/smolagents/task_layer_graph.pkl"

OUTPUT_DIR = "data/results/melo_memory/smolagents/visualizations"
GRAPH_IMAGE_FILE = os.path.join(OUTPUT_DIR, "task_layer_graph.png")

def visualize_sqlite():
    """
    Connects to the SQLite database, prints table names, schemas, and dumps each table to a CSV file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DB_FILE):
        print(f"Database file not found: {DB_FILE}")
        return

    print(f"--- Analyzing SQLite Database: {DB_FILE} ---")
    try:
        with sqlite3.connect(DB_FILE) as con:
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            if not tables:
                print("No tables found in the database.")
                return

            print("\nTables found in the database:")
            table_names = [table[0] for table in tables]
            print(table_names)

            for table_name in table_names:
                print(f"\n--- Table: {table_name} ---")

                # Print schema
                print("Schema:")
                cursor.execute(f"PRAGMA table_info({table_name});")
                schema = cursor.fetchall()
                for row in schema:
                    print(row)

                # Save to CSV
                try:
                    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
                    csv_path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"Data from table '{table_name}' has been saved to {csv_path}")
                except Exception as e:
                    print(f"Could not read table '{table_name}' into pandas or save as CSV: {e}")


    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def visualize_pkl():
    """
    Loads a pickled file, assumes it's a networkx graph, and visualizes it.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(GRAPH_FILE):
        print(f"Pickle file not found: {GRAPH_FILE}")
        return

    print(f"\n--- Visualizing Pickle File: {GRAPH_FILE} ---")
    try:
        with open(GRAPH_FILE, 'rb') as f:
            graph = pickle.load(f)

        if isinstance(graph, nx.Graph):
            print("Pickle file contains a networkx graph.")
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(graph, seed=42)
            nx.draw(graph, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', edge_color='gray', width=0.5)
            plt.title("Task Layer Graph Visualization")
            plt.savefig(GRAPH_IMAGE_FILE)
            plt.close()
            print(f"Graph visualization saved to {GRAPH_IMAGE_FILE}")
        else:
            print(f"The object in the pickle file is not a networkx graph. It is of type: {type(graph)}")

    except pickle.UnpicklingError:
        print("Error: The file is not a valid pickle file or is corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred while processing the pickle file: {e}")

if __name__ == "__main__":
    visualize_sqlite()
    visualize_pkl()
