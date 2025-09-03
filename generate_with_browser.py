#!/usr/bin/env python3
"""
CSV Batch Processor with AI Integration
Processes rows 4-20 from Data.csv and generates AI responses for each prompt template
"""

import argparse
import asyncio
import datetime
import os
import csv
from pathlib import Path

from gpt_oss.tools.simple_browser import SimpleBrowserTool
from gpt_oss.tools.simple_browser.backend import ExaBackend

from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    load_harmony_encoding,
)


REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def create_system_message(args, browser_tool=None):
    """Create system message with optional browser tool"""
    system_message_content = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )
    
    if browser_tool:
        system_message_content = system_message_content.with_tools(browser_tool.tool_config)
    
    return Message.from_role_and_content(Role.SYSTEM, system_message_content)


def detect_encoding(file_path):
    """Detect the encoding of a file"""
    import chardet
    
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return 'utf-8'  # fallback


def read_csv_data(csv_file):
    """Read CSV file with automatic encoding detection and extract data from rows 4-20"""
    csv_data = []
    
    # List of encodings to try
    encodings_to_try = [
        'utf-8',
        'utf-8-sig',  # UTF-8 with BOM
        'iso-8859-1',  # Latin-1
        'cp1252',      # Windows-1252
        'cp850',       # DOS encoding
        'ascii'
    ]
    
    # First try to detect encoding automatically
    detected_encoding = detect_encoding(csv_file)
    if detected_encoding and detected_encoding not in encodings_to_try:
        encodings_to_try.insert(0, detected_encoding)
    elif detected_encoding:
        # Move detected encoding to front
        encodings_to_try.remove(detected_encoding)
        encodings_to_try.insert(0, detected_encoding)
    
    successful_encoding = None
    
    for encoding in encodings_to_try:
        try:
            print(f"Trying encoding: {encoding}")
            with open(csv_file, 'r', encoding=encoding) as f:
                # Try different CSV dialects/delimiters
                sample = f.read(1024)
                f.seek(0)
                
                # Detect delimiter
                sniffer = csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters=',;\t|')
                    delimiter = dialect.delimiter
                    print(f"Detected delimiter: '{delimiter}'")
                except:
                    delimiter = ','  # default fallback
                    print(f"Using default delimiter: ','")
                
                # Read the CSV
                reader = csv.reader(f, delimiter=delimiter)
                all_rows = list(reader)
                successful_encoding = encoding
                print(f"Successfully read CSV with encoding: {encoding}")
                break
                
        except UnicodeDecodeError:
            print(f"Encoding {encoding} failed with UnicodeDecodeError")
            continue
        except Exception as e:
            print(f"Encoding {encoding} failed with error: {e}")
            continue
    
    if successful_encoding is None:
        print("Error: Could not read CSV file with any supported encoding!")
        return []
    
    try:
        # Process rows 4-20 (index 3-19 in 0-based indexing)
        for i in range(3, min(20, len(all_rows))):
            row = all_rows[i]
            if len(row) >= 9:  # Ensure we have columns A through I
                csv_data.append({
                    'A': str(row[0]).strip(),  # Bundesland
                    'E': str(row[4]).strip(),  # Textkennzeichen
                    'H': str(row[7]).strip(),  # Gemeinde/Stadt
                    'I': str(row[8]).strip()   # Verwaltungssitz
                })
            else:
                print(f"Warning: Row {i+1} doesn't have enough columns, skipping...")
                
    except FileNotFoundError:
        print(f"Error: CSV file {csv_file} not found!")
        return []
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return []
    
    print(f"Successfully loaded {len(csv_data)} rows from CSV")
    return csv_data


def load_prompt_templates():
    """Load prompt templates from .txt files"""
    prompt_files = ["prompt1.txt", "prompt2.txt", "prompt3.txt"]
    templates = {}
    
    for prompt_file in prompt_files:
        if os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    templates[prompt_file] = f.read().strip()
                print(f"Loaded template: {prompt_file}")
            except Exception as e:
                print(f"Error loading {prompt_file}: {e}")
        else:
            print(f"Warning: {prompt_file} not found, skipping...")
    
    return templates


def substitute_variables(template, data):
    """Substitute variables in template with actual data"""
    substituted = template.replace("{{A}}", str(data['A']))
    substituted = substituted.replace("{{E}}", str(data['E']))
    substituted = substituted.replace("{{H}}", str(data['H']))
    substituted = substituted.replace("{{I}}", str(data['I']))
    return substituted


def extract_response_text(messages):
    """Extract only the AI response text, filtering out thinking/browsing process"""
    response_text = ""
    
    for message in messages:
        if message.role == Role.ASSISTANT and not message.recipient:
            # This is the final assistant response
            for content in message.content:
                if hasattr(content, 'text'):
                    response_text += content.text
    
    return response_text.strip()


async def process_single_prompt(generator, encoding, prompt_text, args, browser_tool=None):
    """Process a single prompt and return only the response text"""
    try:
        # Limit prompt length to prevent CUDA errors
        if len(prompt_text) > 4000:
            prompt_text = prompt_text[:4000] + "\n\nBitte beantworte diese Anfrage kurz und präzise."
            print("Warning: Prompt truncated to prevent CUDA errors")
        
        # Create fresh message list for each prompt
        system_message = create_system_message(args, browser_tool)
        user_message = Message.from_role_and_content(Role.USER, prompt_text)
        messages = [system_message, user_message]
        
        # Set max tokens to prevent context overflow
        max_response_tokens = 500
        tokens_generated = 0
        
        while True:
            last_message = messages[-1]
            
            # Handle tool calls
            if last_message.recipient and last_message.recipient.startswith("browser."):
                if not browser_tool:
                    raise ValueError("Browser tool is not enabled")
                
                async def run_tool():
                    results = []
                    async for msg in browser_tool.process(last_message):
                        results.append(msg)
                    return results
                
                result = await run_tool()
                messages += result
                continue
            
            # Generate assistant response
            conversation = Conversation.from_messages(messages)
            tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
            
            # Check token count before generation
            if len(tokens) > args.context - 1000:
                print("Warning: Context too long, truncating...")
                return "ERROR: Context too long for processing"
            
            parser = StreamableParser(encoding, role=Role.ASSISTANT)
            current_output_text = ""
            output_text_delta_buffer = ""
            
            for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
                parser.process(predicted_token)
                
                # Check if we've generated too many tokens
                tokens_generated += 1
                if tokens_generated > max_response_tokens:
                    print("Response truncated to prevent overflow")
                    break
                
                if not parser.last_content_delta:
                    continue
                
                should_send_output_text_delta = True
                output_text_delta_buffer += parser.last_content_delta
                
                # Handle browser tool citation normalization
                if browser_tool:
                    updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(
                        current_output_text + output_text_delta_buffer
                    )
                    output_text_delta_buffer = updated_output_text[len(current_output_text):]
                    if has_partial_citations:
                        should_send_output_text_delta = False
                
                if should_send_output_text_delta:
                    current_output_text += output_text_delta_buffer
                    output_text_delta_buffer = ""
            
            messages += parser.messages
            
            # Check if we need to continue with tool calls
            if messages[-1].recipient:
                continue
            else:
                # Assistant response complete
                break
        
        # Extract only the response text
        response_text = extract_response_text(messages)
        return response_text if response_text else "ERROR: Empty response generated"
        
    except Exception as e:
        error_msg = str(e)
        if "CUDA error" in error_msg:
            return f"CUDA_ERROR: {error_msg[:100]}..."
        else:
            return f"ERROR: {error_msg[:100]}..."


def save_results_to_csv(results, output_file="ai_responses.csv"):
    """Save results to a new CSV file"""
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = ['Row', 'Bundesland', 'Textkennzeichen', 'Gemeinde_Stadt', 'Verwaltungssitz']
            for prompt_file in ['prompt1.txt', 'prompt2.txt', 'prompt3.txt']:
                headers.append(f'Response_{prompt_file.replace(".txt", "")}')
            writer.writerow(headers)
            
            # Write data
            for result in results:
                row_data = [
                    result['row_number'],
                    result['data']['A'],
                    result['data']['E'],
                    result['data']['H'],
                    result['data']['I']
                ]
                
                for prompt_file in ['prompt1.txt', 'prompt2.txt', 'prompt3.txt']:
                    response = result.get('responses', {}).get(prompt_file, 'No response')
                    row_data.append(response)
                
                writer.writerow(row_data)
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")


async def main(args):
    # Initialize model backend
    match args.backend:
        case "triton":
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, args.context, device)
        case "torch":
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device)
        case "vllm":
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=2)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Initialize browser tool if enabled
    browser_tool = None
    if args.browser:
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        print("Browser tool enabled")

    # Read CSV data
    print("Reading CSV data...")
    csv_data = read_csv_data("Data.csv")
    if not csv_data:
        print("No data found in CSV file. Exiting.")
        return
    
    print(f"Found {len(csv_data)} rows to process")

    # Load prompt templates
    print("Loading prompt templates...")
    templates = load_prompt_templates()
    if not templates:
        print("No prompt templates found. Exiting.")
        return

    # Process each row
    results = []
    output_file = "ai_responses.csv"
    
    for row_idx, data in enumerate(csv_data, start=4):  # Start from row 4
        print(f"\n{'='*60}")
        print(f"Processing row {row_idx}: {data['H']} ({data['A']})")
        print(f"{'='*60}")
        
        row_result = {
            'row_number': row_idx,
            'data': data,
            'responses': {}
        }
        
        # Process each prompt template for this row
        for template_name, template_content in templates.items():
            print(f"\nProcessing {template_name}...")
            
            # Substitute variables
            prompt_text = substitute_variables(template_content, data)
            print(f"Prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
            
            # Get AI response
            response = await process_single_prompt(generator, encoding, prompt_text, args, browser_tool)
            row_result['responses'][template_name] = response
            
            print(f"Response received: {len(response)} characters")
        
        results.append(row_result)
        
        # Save result immediately after each row is processed
        write_header = (row_idx == 4)  # Write header only for first row
        append_result_to_csv(row_result, output_file, write_header)
    
    # Final save (as backup, in case individual saves failed)
    print(f"\n{'='*60}")
    print("Final backup save...")
    print(f"{'='*60}")
    save_results_to_csv(results, "ai_responses_backup.csv")
    
    print("Processing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV Batch Processor with AI Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
    )
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        metavar="REASONING_EFFORT",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning effort",
    )
    parser.add_argument(
        "-b",
        "--browser",
        default=False,
        action="store_true",
        help="Enable browser tool",
    )
    parser.add_argument(
        "--show-browser-results",
        default=False,
        action="store_true",
        help="Show detailed browser results",
    )
    parser.add_argument(
        "-c",
        "--context",
        metavar="CONTEXT",
        type=int,
        default=8192,
        help="Max context length",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inference backend",
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main(args))
