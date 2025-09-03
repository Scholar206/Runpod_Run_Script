#!/usr/bin/env python3
"""
Harmony chat batch processing with CSV data integration and browser tool support
Processes CSV data through multiple prompt templates
"""

import argparse
import asyncio
import datetime
import os
import csv
import chardet
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


def detect_encoding(file_path):
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def read_csv_data(csv_file):
    """Read CSV file and return data from specified rows"""
    # Detect encoding
    encoding = detect_encoding(csv_file)
    print(f"Detected encoding for {csv_file}: {encoding}")
    
    data_rows = []
    
    with open(csv_file, 'r', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=',')
        all_rows = list(reader)
        
        # Extract rows 4-20 (indices 3-19 in 0-based indexing)
        for row_idx in range(3, min(20, len(all_rows))):
            if row_idx < len(all_rows):
                row = all_rows[row_idx]
                # Extract columns A (0), E (4), H (7), I (8)
                row_data = {
                    'A': row[0] if len(row) > 0 else '',  # Bundesland
                    'E': row[4] if len(row) > 4 else '',  # Textkennzeichen
                    'H': row[7] if len(row) > 7 else '',  # Gemeinde/Stadt
                    'I': row[8] if len(row) > 8 else '',  # Verwaltungssitz
                    'row_number': row_idx + 1  # Keep track of original row number (1-based)
                }
                data_rows.append(row_data)
    
    return data_rows


def replace_variables_in_prompt(prompt_template, row_data):
    """Replace {{A}}, {{E}}, {{H}}, {{I}} variables in prompt template"""
    prompt = prompt_template
    for key in ['A', 'E', 'H', 'I']:
        placeholder = f"{{{{{key}}}}}"
        prompt = prompt.replace(placeholder, row_data[key])
    return prompt


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


def extract_assistant_response(messages):
    """Extract only the assistant's final response text, excluding thinking/browsing"""
    for msg in reversed(messages):
        if msg.role == Role.ASSISTANT and msg.content:
            # Get the last assistant message content
            for content_item in msg.content:
                if hasattr(content_item, 'text'):
                    return content_item.text
    return ""


async def process_single_prompt(generator, encoding, prompt_text, args, browser_tool=None):
    """Process a single prompt with fresh context"""
    
    response_text = ""
    
    try:
        # Create fresh message list for each prompt
        system_message = create_system_message(args, browser_tool)
        user_message = Message.from_role_and_content(Role.USER, prompt_text)
        messages = [system_message, user_message]
        
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
            
            parser = StreamableParser(encoding, role=Role.ASSISTANT)
            current_output_text = ""
            output_text_delta_buffer = ""
            
            for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
                parser.process(predicted_token)
                
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
                # Assistant response complete - extract only the final response
                response_text = extract_assistant_response(messages)
                break
        
        return response_text
        
    except Exception as e:
        print(f"\nError in process_single_prompt: {e}")
        return f"ERROR: {e}"


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
    print(f"Found {len(csv_data)} rows to process (rows 4-20)")

    # Read prompt templates
    prompt_files = ["prompt1.txt", "prompt2.txt", "prompt3.txt"]
    prompt_templates = {}
    
    for prompt_file in prompt_files:
        if not os.path.exists(prompt_file):
            print(f"Warning: {prompt_file} not found, skipping...")
            continue
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_templates[prompt_file] = f.read().strip()
    
    # Prepare output CSV
    output_filename = f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
        # CSV header: Row, Bundesland, Textkennzeichen, Gemeinde, Verwaltungssitz, Prompt1_Response, Prompt2_Response, Prompt3_Response
        fieldnames = ['Row', 'Bundesland', 'Textkennzeichen', 'Gemeinde', 'Verwaltungssitz']
        for i in range(1, 4):
            fieldnames.append(f'Prompt{i}_Response')
        
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each data row
        for row_idx, row_data in enumerate(csv_data, 1):
            print(f"\n{'='*60}")
            print(f"Processing row {row_data['row_number']}: {row_data['H']} ({row_data['A']})")
            print(f"{'='*60}")
            
            # Prepare output row
            output_row = {
                'Row': row_data['row_number'],
                'Bundesland': row_data['A'],
                'Textkennzeichen': row_data['E'],
                'Gemeinde': row_data['H'],
                'Verwaltungssitz': row_data['I']
            }
            
            # Process each prompt template for this row
            for prompt_num, (prompt_file, prompt_template) in enumerate(prompt_templates.items(), 1):
                print(f"\nProcessing {prompt_file} for row {row_data['row_number']}...")
                
                # Replace variables in prompt
                filled_prompt = replace_variables_in_prompt(prompt_template, row_data)
                
                # Process the prompt
                response = await process_single_prompt(generator, encoding, filled_prompt, args, browser_tool)
                
                # Store only the response (not the thinking/browsing process)
                output_row[f'Prompt{prompt_num}_Response'] = response
                
                print(f"Response captured for {prompt_file}")
            
            # Write row to CSV
            writer.writerow(output_row)
            output_file.flush()  # Ensure data is written immediately
            
            print(f"Row {row_data['row_number']} completed and saved")

    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_filename}")
    print(f"Processed {len(csv_data)} rows with {len(prompt_templates)} prompts each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSV batch processing with browser tool support",
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
