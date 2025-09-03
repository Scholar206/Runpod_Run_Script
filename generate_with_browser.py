#!/usr/bin/env python3
"""
Batch generation with browser tool support
Processes multiple prompt files sequentially with fresh context for each prompt
"""

import argparse
import asyncio
import datetime
import os
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


async def process_single_prompt(generator, encoding, prompt_text, args, browser_tool=None):
    """Process a single prompt with fresh context"""
    print(f"\n{'='*60}")
    print(f"Processing prompt: {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
    print(f"{'='*60}")
    
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
            
            print("\n[Browser tool executing...]")
            
            async def run_tool():
                results = []
                async for msg in browser_tool.process(last_message):
                    results.append(msg)
                return results
            
            result = await run_tool()
            messages += result
            
            if not args.show_browser_results:
                print("[Browser results processed]")
            else:
                print(f"Browser output: {result[0].content[0].text}")
            
            continue
        
        # Generate assistant response
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        current_output_text = ""
        output_text_delta_buffer = ""
        
        print("\nAssistant Response:")
        print("-" * 40)
        
        for predicted_token in generator.generate(
            tokens, 
            encoding.stop_tokens_for_assistant_actions(),
            temperature=args.temperature,
            max_tokens=args.max_tokens if args.max_tokens > 0 else None
        ):
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
                print(output_text_delta_buffer, end="", flush=True)
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""
        
        messages += parser.messages
        
        # Check if we need to continue with tool calls
        if messages[-1].recipient:
            continue
        else:
            # Assistant response complete
            break
    
    print("\n" + "-" * 40)
    print("Response complete\n")


async def main(args):
    # Initialize model backend
    match args.backend:
        case "triton":
            from gpt_oss.triton.model import TokenGenerator as TritonGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TritonGenerator(args.checkpoint, args.context_length, device)
        case "torch":
            from gpt_oss.torch.model import TokenGenerator as TorchGenerator
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            generator = TorchGenerator(args.checkpoint, device)
        case "vllm":
            from gpt_oss.vllm.token_generator import TokenGenerator as VLLMGenerator
            generator = VLLMGenerator(args.checkpoint, tensor_parallel_size=args.tensor_parallel_size)
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Initialize browser tool if enabled
    browser_tool = None
    if args.browser:
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        print("Browser tool enabled")

    # Process prompt files
    prompt_files = ["prompt1.txt", "prompt2.txt", "prompt3.txt"]
    
    for prompt_file in prompt_files:
        if not os.path.exists(prompt_file):
            print(f"Warning: {prompt_file} not found, skipping...")
            continue
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            
            if not prompt_text:
                print(f"Warning: {prompt_file} is empty, skipping...")
                continue
            
            await process_single_prompt(generator, encoding, prompt_text, args, browser_tool)
            
        except Exception as e:
            print(f"Error processing {prompt_file}: {e}")
            continue

    print("All prompts processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generation with browser tool support",
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
        "-t",
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        metavar="LIMIT",
        type=int,
        default=0,
        help="Maximum tokens to generate (0 for unlimited)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inference backend",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM backend",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        help="Context length for Triton backend",
    )
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main(args))
