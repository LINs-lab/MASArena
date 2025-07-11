# This code is adapted from the Agents_Failure_Attribution project
# Original repository: https://github.com/microsoft/autogen/tree/main/notebook/agentchat_contrib
# We acknowledge the original authors and contributors for their work

import argparse
import contextlib
import sys
import os
import datetime
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

from lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search,
    convert_txt_to_json,
    generate_timestamped_filename
)


KNOWN_GPT_MODELS = {"gpt-4o", "gpt-4", "gpt-4o-mini", "gpt-4.1"}
ALL_MODELS = list(KNOWN_GPT_MODELS)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent responses for failure attribution.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"GPT model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default="../../results/agent_responses",
        help="Path to the directory containing agent response JSON files. Default: '../../results/agent_responses'."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../results/failure",
        help="Path to the output directory for failure attribution results. Default: '../../results/failure'."
    )

    parser.add_argument(
        "--api_key", type=str, default="",
        help="OpenAI API Key. Uses OPENAI_API_KEY or AZURE_OPENAI_API_KEY env var if available."
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default="",
        help="Azure OpenAI Endpoint URL. If not provided, will use standard OpenAI API. Uses AZURE_OPENAI_ENDPOINT env var if available."
    )
    parser.add_argument(
        "--openai_base_url", type=str, default="",
        help="OpenAI API Base URL. Uses OPENAI_API_BASE env var if available."
    )
    parser.add_argument(
        "--api_version", type=str, default="2024-08-01-preview",
        help="Azure OpenAI API Version."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response."
    )

    args = parser.parse_args()

    # Get API credentials from environment if not provided
    if not args.api_key:
        args.api_key = os.getenv('OPENAI_API_KEY', '') or os.getenv('AZURE_OPENAI_API_KEY', '')
    if not args.azure_endpoint:
        args.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    if not args.openai_base_url:
        args.openai_base_url = os.getenv('OPENAI_API_BASE', '')

    if args.model not in KNOWN_GPT_MODELS:
        print(f"Error: Invalid model '{args.model}' specified. Only GPT models are supported.")
        sys.exit(1)

    print(f"Selected GPT model: {args.model}")
   
    if not args.api_key:
        print("Error: --api_key or OPENAI_API_KEY/AZURE_OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    try:
        if args.azure_endpoint:
            # Use Azure OpenAI
            client = AzureOpenAI(
                api_key=args.api_key,
                api_version=args.api_version,
                azure_endpoint=args.azure_endpoint,
            )
            print(f"Successfully initialized AzureOpenAI client for endpoint: {args.azure_endpoint}")
        else:
            # Use standard OpenAI API
            openai_kwargs = {"api_key": args.api_key}
            if args.openai_base_url:
                openai_kwargs["base_url"] = args.openai_base_url
                print(f"Using custom OpenAI base URL: {args.openai_base_url}")
            
            client = OpenAI(**openai_kwargs)
            print(f"Successfully initialized OpenAI client")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamped filenames
    base_filename = f"{args.method}_{args.model.replace('/','_')}_agent_responses"
    txt_filename = generate_timestamped_filename(base_filename, "txt")
    json_filename = generate_timestamped_filename(base_filename, "json")
    
    txt_filepath = os.path.join(output_dir, txt_filename)
    json_filepath = os.path.join(output_dir, json_filename)

    print(f"Analysis method: {args.method}")
    print(f"Model: {args.model}")
    print(f"TXT output will be saved to: {txt_filepath}")
    print(f"JSON output will be saved to: {json_filepath}")

    try:
        with open(txt_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Used: {args.model}")
            print(f"Input Directory: {args.directory_path}")
            print("-" * 20)

            if args.method == "all_at_once":
                gpt_all_at_once(
                    client=client,
                    directory_path=args.directory_path,
                    model=args.model,
                    max_tokens=args.max_tokens
                )
            elif args.method == "step_by_step":
                gpt_step_by_step(
                    client=client,
                    directory_path=args.directory_path,
                    model=args.model,
                    max_tokens=args.max_tokens
                )
            elif args.method == "binary_search":
                gpt_binary_search(
                    client=client,
                    directory_path=args.directory_path,
                    model=args.model,
                    max_tokens=args.max_tokens
                )

            print("-" * 20)
            print(f"--- Analysis Complete ---")

        # Convert TXT output to JSON format
        print(f"Converting analysis results to JSON format...")
        convert_txt_to_json(
            txt_filepath=txt_filepath,
            json_filepath=json_filepath,
            method=args.method,
            model=args.model,
            directory_path=args.directory_path
        )
        
        print(f"Analysis finished. Outputs saved to:")
        print(f"  TXT: {txt_filepath}")
        print(f"  JSON: {json_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()