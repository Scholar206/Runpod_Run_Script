#!/usr/bin/env python3
"""
Batch-Verarbeitung von Prompts mit CSV-Daten
Liest Data.csv (Zeilen 4–20), ersetzt Variablen in Prompt-Templates
und speichert die Modellantworten in einer neuen CSV-Datei.
"""

import argparse
import asyncio
import datetime
import os
from pathlib import Path

import pandas as pd
import chardet

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

# ---------------------------
# Hilfsfunktionen
# ---------------------------

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}


def detect_encoding(file_path: str) -> str:
    """Bestimme Encoding einer Datei"""
    with open(file_path, "rb") as f:
        result = chardet.detect(f.read(50000))
    print(f"[INFO] Encoding erkannt: {result}")
    return result["encoding"]


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
    """Verarbeitet einen einzelnen Prompt und gibt nur die Antwort zurück"""
    try:
        # Create fresh message list
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
                
                output_text_delta_buffer += parser.last_content_delta
                
                # Normalisiere Browser-Tool-Citations
                if browser_tool:
                    updated_output_text, _annotations, has_partial_citations = browser_tool.normalize_citations(
                        current_output_text + output_text_delta_buffer
                    )
                    output_text_delta_buffer = updated_output_text[len(current_output_text):]
                    if has_partial_citations:
                        continue
                
                current_output_text += output_text_delta_buffer
                output_text_delta_buffer = ""
            
            messages += parser.messages
            
            if messages[-1].recipient:
                continue
            else:
                # Antwort fertig
                break
        
        return current_output_text.strip()
    
    except Exception as e:
        print(f"\nError in process_single_prompt: {e}")
        return f"ERROR: {e}"


def fill_prompt(template_path, variables):
    """Lade Prompt-Template und ersetze {{Variablen}}"""
    with open(template_path, "r", encoding="utf-8") as f:
        text = f.read()
    for key, value in variables.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


# ---------------------------
# Hauptprogramm
# ---------------------------

async def main(args):
    # Backend laden
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

    # Browser-Tool optional aktivieren
    browser_tool = None
    if args.browser:
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        print("Browser tool enabled")

    # CSV einlesen
    encoding_csv = detect_encoding("Data.csv")
    df = pd.read_csv("Data.csv", encoding=encoding_csv, sep=";")
    df_subset = df.iloc[3:20, [0, 4, 7, 8]]
    df_subset.columns = ["A", "E", "H", "I"]

    results = []

    # Prompts verarbeiten
    for idx, row in df_subset.iterrows():
        variables = {"A": row["A"], "E": row["E"], "H": row["H"], "I": row["I"]}

        for prompt_file in ["prompt1.txt", "prompt2.txt", "prompt3.txt"]:
            if not os.path.exists(prompt_file):
                continue

            prompt_text = fill_prompt(prompt_file, variables)
            response_text = await process_single_prompt(generator, encoding, prompt_text, args, browser_tool)

            results.append({
                "Zeile": idx + 1,
                "Prompt": prompt_file,
                "Antwort": response_text
            })

    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv("Model_Antworten.csv", encoding="utf-8", index=False)
    print("[INFO] Ergebnisse gespeichert in Model_Antworten.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generation with CSV + Prompt-Templates",
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
    
    asyncio.run(main(args))
