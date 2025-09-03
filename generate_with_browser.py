#!/usr/bin/env python3
"""
Batch‑Processing für gpt‑OSS – CSV‑Einlesen, Prompt‑Ersetzung und Modell‑Antwort
speichern.
Der Code ist auf den in der offiziellen Repository‑Dokumentation (gh‑openai/gpt-oss)
beschriebenen Funktionen & Importen vollständig anwendbar.

Author:  OpenAI gpt-OSS team
Date:    2025‑09‑03
"""

import argparse
import asyncio
import csv
import datetime
import os
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# gpt‑OSS Imports – exakt wie im Haupt‑chat‑Skript
# ────────────────────────────────────────────────────────────────────────
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
    SystemContent,
    TextContent,
    load_harmony_encoding,
)

# ------------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------------
REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

# ------------------------------------------------------------------
# System‑Nachricht – exakt wie im original chat.py
# ------------------------------------------------------------------
def create_system_message(args, browser_tool=None):
    content = SystemContent.new() \
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort]) \
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    if browser_tool:
        content = content.with_tools(browser_tool.tool_config)
    return Message.from_role_and_content(Role.SYSTEM, content)


# ------------------------------------------------------------------
# Prompt‑Verarbeitung – liefert nur *reinen* Modell‑Output zurück
# ------------------------------------------------------------------
async def process_single_prompt(
    generator, encoding, prompt_text, args, browser_tool=None
) -> str:
    """
    Der Prompt wird ausgeführt und nur der endgültige Text des Modells (keine
    Tool‑Calls, keine Browser‑Citations) zurückgegeben.
    """
    system_message = create_system_message(args, browser_tool)
    user_message = Message.from_role_and_content(Role.USER, prompt_text)

    messages = [system_message, user_message]
    assistant_output = ""

    while True:
        last_message = messages[-1]

        # -------- Browser‑Tool ausführen (falls vorhanden) --------
        if last_message.recipient and last_message.recipient.startswith("browser."):
            if not browser_tool:
                raise ValueError("Browser‑Tool ist nicht aktiviert")
            async def run_tool():
                results = []
                async for msg in browser_tool.process(last_message):
                    results.append(msg)
                return results
            result = await run_tool()
            messages += result
            continue

        # -------- Modellantwort generieren --------
        conversation = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        output_buffer = ""

        for predicted_token in generator.generate(tokens, encoding.stop_tokens_for_assistant_actions()):
            parser.process(predicted_token)
            if not parser.last_content_delta:
                continue

            output_buffer += parser.last_content_delta

            # Citation‑Normalisierung (nur im Browser‑Modus nötig)
            if browser_tool:
                updated_text, _ann, has_partial = browser_tool.normalize_citations(
                    assistant_output + output_buffer
                )
                output_buffer = updated_text[len(assistant_output) :]
                if has_partial:
                    continue

            assistant_output += output_buffer
            output_buffer = ""

        messages += parser.messages

        # -------- Weiterführen, wenn Tool‑Calls offen -----
        if messages[-1].recipient:
            continue

        break

    return assistant_output.strip()


# ------------------------------------------------------------------
# Hauptfunktion – CSV‑Einlesen, Prompt‑Lösung, CSV‑Ausgabe
# ------------------------------------------------------------------
async def main(args):
    # 1️⃣ Backend‑Initialisierung – identisch zu chat.py
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
            raise ValueError(f"Ungültiger Backend‑Typ: {args.backend}")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # 2️⃣ Optionaler Browser‑Tool
    browser_tool = None
    if args.browser:
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        print("Browser‑Tool aktiviert")

    # 3️⃣ CSV‑Datei einlesen (ISO‑8859‑9) – Zeilen 4‑20
    csv_path = Path("Data.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} nicht gefunden")

    with open(csv_path, encoding="iso-8859-9") as f:
        reader = list(csv.reader(f))

    header = reader[0]            # nicht zwingend nötig, aber hilfreich
    data_rows = reader[3:20]      # 4‑20 (Index 3‑19)

    # 4️⃣ Prompt‑Templates im Speicher halten
    prompt_templates: dict[str, str] = {}
    for pn in ["prompt1", "prompt2", "prompt3"]:
        fn = f"{pn}.txt"
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as fp:
                prompt_templates[pn] = fp.read()
        else:
            print(f"Warnung: {fn} nicht gefunden – ignoriere")

    # 5️⃣ Zeilen durchlaufen, Platzhalter ersetzen und Prompt ausführen
    output_rows = []
    for line_no, row in enumerate(data_rows, start=4):
        record = {
            "line": str(line_no),
            "A": row[0] if len(row) > 0 else "",
            "E": row[4] if len(row) > 4 else "",
            "H": row[7] if len(row) > 7 else "",
            "I": row[8] if len(row) > 8 else "",
        }

        answers: dict[str, str] = {}
        for pn, tmpl in prompt_templates.items():
            prompt = tmpl.replace("{{A}}", record["A"]) \
                         .replace("{{E}}", record["E"]) \
                         .replace("{{H}}", record["H"]) \
                         .replace("{{I}}", record["I"])

            answer = await process_single_prompt(
                generator, encoding, prompt, args, browser_tool
            )
            answers[pn] = answer

        output_rows.append(
            {
                "line": record["line"],
                "A": record["A"],
                "E": record["E"],
                "H": record["H"],
                "I": record["I"],
                "prompt1": answers.get("prompt1", ""),
                "prompt2": answers.get("prompt2", ""),
                "prompt3": answers.get("prompt3", ""),
            }
        )

    # 6️⃣ Ergebnisse in neue CSV schreiben
    out_path = Path("output.csv")
    fieldnames = ["line", "A", "E", "H", "I", "prompt1", "prompt2", "prompt3"]
    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nAlle Antworten wurden in {out_path} gespeichert.")


# ------------------------------------------------------------------
# Argumente parsen & Start
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch‑Generierung mit optionalem Browser‑Tool (basierend auf chat.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str, help="Pfad zur SafeTensors‑Checkpoint‑Datei")
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["high", "medium", "low"],
        help="Reasoning‑Effort des Modells",
    )
    parser.add_argument("-b", "--browser", action="store_true", help="Browser‑Tool aktivieren")
    parser.add_argument(
        "--show-browser-results",
        action="store_true",
        help="Browser‑Ergebnisse im Detail anzeigen",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=int,
        default=8192,
        help="Maximale Kontext‑Länge",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inferenz‑Backend wählen",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
