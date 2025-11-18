#!/usr/bin/env python3
"""
Utility script to visualize benchmark results from saved summary and results files.

This script provides a command-line interface to generate visualizations
for benchmark results, including links to individual problem visualizations.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from mas_arena.visualization.mas_visualizer import BenchmarkVisualizer
except ImportError:
    print("Could not import BenchmarkVisualizer. Make sure you're running this script from the project root.")
    sys.exit(1)


def list_available_files(args):
    """List available summary and results files"""
    summaries_dir = args.summaries_dir or "results"
    results_dir = args.results_dir or "results"
    
    summaries_dir = Path(summaries_dir)
    results_dir = Path(results_dir)
    
    # Find all summary files
    summary_files = list(summaries_dir.glob("*_summary.json"))
    
    # Find all results files
    results_files = list(results_dir.glob("*.json"))
    results_files = [f for f in results_files if not f.name.endswith("_summary.json")]
    
    # Filter by agent system if provided
    if args.agent_system:
        summary_files = [f for f in summary_files if args.agent_system in f.name]
        results_files = [f for f in results_files if args.agent_system in f.name]
    
    # Filter by benchmark if provided
    if args.benchmark:
        summary_files = [f for f in summary_files if args.benchmark in f.name]
        results_files = [f for f in results_files if args.benchmark in f.name]
    
    print(f"Found {len(summary_files)} summary files:")
    for i, file in enumerate(summary_files):
        print(f"  [{i}] {file}")
    
    print(f"\nFound {len(results_files)} results files:")
    for i, file in enumerate(results_files):
        print(f"  [{i}] {file}")
    
    return summary_files, results_files


def visualize_benchmark(args):
    """Visualize a benchmark summary file"""
    visualizer = BenchmarkVisualizer(args.output_dir)
    
    if args.summary:
        summary_file = Path(args.summary)
        if summary_file.exists():
            # Determine results file if not specified
            results_file = args.results
            if not results_file:
                # Try to infer from summary data
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    if "results_file" in summary_data:
                        results_file = summary_data["results_file"]
            
            # Generate visualization
            visualizer.visualize_benchmark(
                summary_file=str(summary_file),
                results_file=results_file,
                visualizations_dir=args.visualizations_dir,
                output_file=args.output
            )
        else:
            print(f"Summary file not found: {summary_file}")
    elif args.index is not None:
        # List files and visualize by index
        summary_files, results_files = list_available_files(args)
        
        if args.index < len(summary_files):
            summary_file = summary_files[args.index]
            
            # Try to find matching results file
            results_file = None
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
                if "results_file" in summary_data:
                    results_file = summary_data["results_file"]
                    
            visualizer.visualize_benchmark(
                summary_file=str(summary_file),
                results_file=results_file,
                visualizations_dir=args.visualizations_dir,
                output_file=args.output
            )
        else:
            print(f"Invalid index: {args.index}")
    else:
        print("Please specify a summary file or index to visualize")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results from saved files")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List available files
    list_parser = subparsers.add_parser("list", help="List available summary and results files")
    list_parser.add_argument("--summaries-dir", help="Directory containing summary files", default="results")
    list_parser.add_argument("--results-dir", help="Directory containing results files", default="results")
    list_parser.add_argument("--agent-system", help="Filter by agent system name")
    list_parser.add_argument("--benchmark", help="Filter by benchmark name")
    
    # Visualize a benchmark
    viz_parser = subparsers.add_parser("visualize", help="Visualize a benchmark summary")
    viz_parser.add_argument("--summary", "-s", help="Path to the benchmark summary file")
    viz_parser.add_argument("--results", "-r", help="Path to the benchmark results file")
    viz_parser.add_argument("--index", "-i", type=int, help="Index of the summary file to visualize (from list command)")
    viz_parser.add_argument("--visualizations-dir", "-v", help="Directory containing problem visualizations", default="results/visualizations")
    viz_parser.add_argument("--output", "-o", help="Path to save the output HTML file")
    viz_parser.add_argument("--output-dir", "-d", help="Directory to save the output HTML file", default="results/visualizations/html")
    viz_parser.add_argument("--summaries-dir", help="Directory containing summary files", default="results")
    viz_parser.add_argument("--results-dir", help="Directory containing results files", default="results")
    viz_parser.add_argument("--agent-system", help="Filter by agent system name")
    viz_parser.add_argument("--benchmark", help="Filter by benchmark name")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_available_files(args)
    elif args.command == "visualize":
        visualize_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 