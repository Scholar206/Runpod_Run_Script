#!/usr/bin/env python3
"""
Batch‑Processing mit gpt‑OSS – Debug‑Ausgaben (Browser‑Tool + High Reasoning + Triton)
"""

import argparse
import asyncio
import csv
import datetime
import os
from pathlib import Path

# ------------------------------------------------------------------
# gpt‑OSS Imports (wie im Original‑Repository)
# ------------------------------------------------------------------
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
    load_harmony_encoding,
)

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

# ------------------------------------------------------------------
# Hilfsfunktion für System‑Nachricht (identisch wie im original chat.py)
# ------------------------------------------------------------------
def create_system_message(args, browser_tool=None):
    sys_msg = (
        SystemContent.new()
        .with_reasoning_effort(REASONING_EFFORT[args.reasoning_effort])
        .with_conversation_start_date(datetime.datetime.now().strftime("%Y-%m-%d"))
    )
    if browser_tool:
        sys_msg = sys_msg.with_tools(browser_tool.tool_config)
    return Message.from_role_and_content(Role.SYSTEM, sys_msg)


# ------------------------------------------------------------------
# Prompt‑Verarbeitung – gibt nur reinen Text zurück
# ------------------------------------------------------------------
async def process_single_prompt(
    generator, encoding, prompt_text, args, browser_tool=None
) -> str:
    print(f"\n=== Verarbeitung Prompt ===\n{prompt_text[:120]}...")

    system_message = create_system_message(args, browser_tool)
    user_message = Message.from_role_and_content(Role.USER, prompt_text)
    messages = [system_message, user_message]
    assistant_output = ""

    while True:
        last_message = messages[-1]

        # ---------- Browser‑Tool ----------
        if last_message.recipient and last_message.recipient.startswith("browser."):
            print("\n[Browser] Aufruf von:", last_message.content[0].text[:60])
            if not browser_tool:
                raise ValueError("Browser‑Tool ist nicht aktiviert")

            async def run_tool():
                results = []
                async for msg in browser_tool.process(last_message):
                    results.append(msg)
                return results

            result = await run_tool()
            messages += result
            print("[Browser] Ergebnis empfangen, weiter geht’s.")
            continue

        # ---------- Modellausgabe ----------
        conv = Conversation.from_messages(messages)
        tokens = encoding.render_conversation_for_completion(conv, Role.ASSISTANT)
        print(f"\n[Model] Tokens vorbereitet: {len(tokens)}")
        parser = StreamableParser(encoding, role=Role.ASSISTANT)
        output_buffer = ""

        for predicted_token in generator.generate(
            tokens, encoding.stop_tokens_for_assistant_actions()
        ):
            parser.process(predicted_token)
            if not parser.last_content_delta:
                continue

            output_buffer += parser.last_content_delta

            if browser_tool:
                updated, _, partial = browser_tool.normalize_citations(
                    assistant_output + output_buffer
                )
                output_buffer = updated[len(assistant_output) :]
                if partial:
                    continue

            assistant_output += output_buffer
            output_buffer = ""

        messages += parser.messages

        # ---------- Prüfung auf weitere Tool‑Calls ----------
        if messages[-1].recipient:
            print("[Debug] Tool‑Call offen – schau nach.")
            continue

        print("\n[Model] Antwort fertig.")
        break

    print("\n=== Ende Prompt ===")
    return assistant_output.strip()


# ------------------------------------------------------------------
# Main‑Funktion (CSV‑Einlesen, Prompt‑Ausführung, CSV‑Ausgabe)
# ------------------------------------------------------------------
async def main(args):
    # 1️⃣ Backend‑Initialisierung
    print("\n[Setup] Backend starten…")
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

    # 2️⃣ Browser‑Tool optional
    browser_tool = None
    if args.browser:
        print("[Setup] Browser‑Tool initialisieren…")
        backend = ExaBackend(source="web")
        browser_tool = SimpleBrowserTool(backend=backend)
        print("[Setup] Browser‑Tool aktiviert")
    else:
        print("[Setup] Browser‑Tool _nicht_ aktiviert")

    # 3️⃣ CSV‑Datei einlesen (ISO‑8859‑9)
    csv_path = Path("Data.csv")
    print(f"\n[Setup] Lies CSV von {csv_path}")
    with open(csv_path, encoding="iso-8859-9") as f:
        reader = list(csv.reader(f))
    data_rows = reader[3:20]  # Zeilen 4‑20
    print(f"[Setup] {len(data_rows)} Zeilen gefunden")

    # 4️⃣ Prompt‑Templates laden
    prompt_templates: dict[str, str] = {}
    for pn in ["prompt1", "prompt2", "prompt3"]:
        fn = f"{pn}.txt"
        if os.path.exists(fn):
            with open(fn, encoding="utf-8") as fp:
                prompt_templates[pn] = fp.read()
            print(f"[Setup] Lade {fn} ({len(prompt_templates[pn])} Zeichen)")
        else:
            print(f"[Warn] {fn} nicht gefunden – ignoriere")

    # 5️⃣ Zeilen durchlaufen
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
        print(f"\n[Zeile {line_no}] {record['A'][:30]} …")
        for pn, tmpl in prompt_templates.items():
            prompt = tmpl.replace("{{A}}", record["A"]).replace(
                "{{E}}", record["E"]
            ).replace("{{H}}", record["H"]).replace("{{I}}", record["I"])

            ans = await process_single_prompt(
                generator, encoding, prompt, args, browser_tool
            )
            answers[pn] = ans
            print(f"[Antwort | {pn}] {ans[:50]}…")

        output_rows.append(
            {**record, "prompt1": answers.get("prompt1", ""), "prompt2": answers.get("prompt2", ""), "prompt3": answers.get("prompt3", "")}
        )

    # 6️⃣ Ergebnisse in CSV schreiben
    out_path = Path("output.csv")
    fieldnames = ["line", "A", "E", "H", "I", "prompt1", "prompt2", "prompt3"]
    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n✅ Alle Antworten wurden in {out_path} gespeichert.")


# ------------------------------------------------------------------
# Argument‑Parsing
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch‑Generierung mit Browser‑Tool (triton‑backend, high‑reasoning).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="Pfad zur SafeTensors‑Checkpoint‑Datei")
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        default="high",
        choices=["high", "medium", "low"],
        help="Reasoning‑Effort des Modells",
    )
    parser.add_argument(
        "-b",
        "--browser",
        action="store_true",
        help="Browser‑Tool aktivieren",
    )
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
        default="triton",
        choices=["triton", "torch", "vllm"],
        help="Inferenz‑Backend wählen",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
