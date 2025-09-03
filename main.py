import subprocess
import sys
import time
import os
from typing import List, Dict
import threading
import queue
import re
import select

class GPTOSSPromptSender:
    def __init__(self, model_path: str = "gpt-oss-120b/original/"):
        self.model_path = model_path
        self.process = None
        self.output_queue = queue.Queue()
        self.stop_reading = False
        
        # Drei kleine Test-Prompts (statt der langen Hochwasserschutz-Templates)
        self.prompts = [
            """# PROMPT 1: INTERNE RECHERCHE (ALLGEMEIN)

Führe eine interne Recherche durch und beantworte:
- Was ist die Hauptstadt von Deutschland?  

## AUSGABEFORMAT
Schreibe nur den Stadtnamen, sonst nichts.
""",

            """# PROMPT 2: INTERNE RECHERCHE (EINFACH)

Führe eine interne Recherche durch und beantworte:
- Welche Sprache wird in Deutschland hauptsächlich gesprochen?  

## AUSGABEFORMAT
Nur die Sprache nennen.
""",

            """# PROMPT 3: INTERNE RECHERCHE (KURZ)

Führe eine interne Recherche durch und beantworte:
- Nenne die ungefähre Einwohnerzahl von Darmstadt.  

## AUSGABEFORMAT
Nur die Zahl mit Einheit (z. B. '50.000 Einwohner').
"""
        ]

    def start_gpt_process(self):
        """Startet den GPT-OSS Prozess einmalig"""
        # Pfad zum Parent-Verzeichnis berechnen
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        cmd = [
            "python", "-m", "gpt_oss.chat",
            self.model_path,
            "--reasoning-effort", "high",
            "--browser",
            "--python", 
            "--show-browser-results",
            "--context", "2048"
        ]
        
        print(f"Starte GPT-OSS Prozess im Parent-Verzeichnis: {parent_dir}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # Unbepuffert für bessere Kontrolle
            universal_newlines=True,
            cwd=parent_dir
        )
        
        # Starte Output-Reader Thread
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()
        
        # Warte auf Initialisierung
        print("Warte auf Modell-Initialisierung...")
        self._wait_for_ready()

    def _read_output(self):
        """Liest kontinuierlich Output vom Prozess"""
        try:
            while not self.stop_reading and self.process.poll() is None:
                if hasattr(select, 'select'):
                    ready, _, _ = select.select([self.process.stdout], [], [], 0.1)
                    if ready:
                        line = self.process.stdout.readline()
                        if line:
                            self.output_queue.put(line)
                else:
                    try:
                        line = self.process.stdout.readline()
                        if line:
                            self.output_queue.put(line)
                    except:
                        time.sleep(0.1)
                        continue
        except Exception as e:
            print(f"Fehler beim Lesen der Ausgabe: {e}")
    
    def _wait_for_ready(self):
        """Wartet bis das System bereit ist"""
        ready_indicators = ["User:", "System Message:", "Model Identity:"]
        timeout = 45  # Erhöht auf 45 Sekunden
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=2)
                print(f"Init: {line.strip()}")
                if any(indicator in line for indicator in ready_indicators):
                    print("System ist bereit!")
                    # Zusätzliche Pause nach Bereitschaft
                    time.sleep(2)
                    return True
            except queue.Empty:
                continue
        
        print("Timeout beim Warten auf System-Bereitschaft")
        return False
    
    def send_prompt(self, prompt: str, row_num: int, prompt_num: int) -> str:
        """Sendet einen Prompt und wartet auf Antwort"""
        if not self.process or self.process.poll() is not None:
            raise Exception("GPT Prozess nicht aktiv")
            
        try:
            print(f"\n=== Sende Prompt {prompt_num} für Zeile {row_num} ===")
            print(f"Prompt Länge: {len(prompt)} Zeichen")
            
            # LÖSUNG 1: Prompt als ein Block mit speziellem Delimiter senden
            delimiter = "<<<END_PROMPT>>>"
            full_message = f"{prompt}\n{delimiter}\n"
            
            # Debug: Zeige ersten Teil des Prompts
            print(f"Erste 200 Zeichen: {prompt[:200]}...")
            
            # Sende den kompletten Prompt mit Delimiter
            self.process.stdin.write(full_message)
            self.process.stdin.flush()
            
            response = self._collect_response()
            print(f"Antwort erhalten für Zeile {row_num}, Prompt {prompt_num}")
            return response
            
        except Exception as e:
            print(f"Fehler beim Senden des Prompts: {e}")
            return f"FEHLER: {e}"
    
    def send_prompt_alternative(self, prompt: str, row_num: int, prompt_num: int) -> str:
        """Alternative Methode: Prompt in Teilen senden"""
        if not self.process or self.process.poll() is not None:
            raise Exception("GPT Prozess nicht aktiv")
            
        try:
            print(f"\n=== Sende Prompt {prompt_num} für Zeile {row_num} (Alternative) ===")
            
            # LÖSUNG 2: Mehrzeilige Prompts korrekt formatieren
            lines = prompt.split('\n')
            for i, line in enumerate(lines):
                if i == len(lines) - 1:  # Letzte Zeile
                    self.process.stdin.write(line + "\n\n")  # Doppeltes Newline am Ende
                else:
                    self.process.stdin.write(line + "\n")
                
            self.process.stdin.flush()
            
            response = self._collect_response()
            print(f"Antwort erhalten für Zeile {row_num}, Prompt {prompt_num}")
            return response
            
        except Exception as e:
            print(f"Fehler beim Senden des Prompts: {e}")
            return f"FEHLER: {e}"
    
    def send_prompt_encoded(self, prompt: str, row_num: int, prompt_num: int) -> str:
        """LÖSUNG 3: Base64-kodierte Übertragung für mehrzeilige Prompts"""
        if not self.process or self.process.poll() is not None:
            raise Exception("GPT Prozess nicht aktiv")
            
        try:
            import base64
            
            print(f"\n=== Sende Prompt {prompt_num} für Zeile {row_num} (Encoded) ===")
            
            # Kodiere den Prompt
            encoded_prompt = base64.b64encode(prompt.encode('utf-8')).decode('ascii')
            
            # Sende Dekodierungsanweisung + kodierten Prompt
            wrapper = f"""Dekodiere den folgenden Base64-kodierten Text und bearbeite ihn als Prompt:

{encoded_prompt}

Dekodiere zuerst mit Base64 und führe dann die Anweisungen aus."""

            self.process.stdin.write(wrapper + "\n")
            self.process.stdin.flush()
            
            response = self._collect_response()
            print(f"Antwort erhalten für Zeile {row_num}, Prompt {prompt_num}")
            return response
            
        except Exception as e:
            print(f"Fehler beim Senden des Prompts: {e}")
            return f"FEHLER: {e}"
    
    def _collect_response(self) -> str:
        """Sammelt die vollständige Antwort"""
        response_lines = []
        assistant_started = False
        response_complete = False
        timeout = 180
        start_time = time.time()
        
        while not response_complete and time.time() - start_time < timeout:
            try:
                line = self.output_queue.get(timeout=2)
                print(f"[DEBUG] {line.strip()}")
                
                if "Assistant:" in line:
                    assistant_started = True
                    continue
                    
                if assistant_started:
                    # Prüfe auf Antwort-Ende
                    if self._is_response_complete(line, response_lines):
                        response_complete = True
                        break
                    if "User:" in line:
                        response_complete = True
                        break
                    response_lines.append(line.rstrip())
                    
            except queue.Empty:
                if assistant_started and response_lines:
                    response_complete = True
                    break
                continue
        
        if not response_complete:
            print("Timeout beim Warten auf vollständige Antwort")
        
        return "\n".join(response_lines)
    
    def _is_response_complete(self, line: str, response_lines: List[str]) -> bool:
        """Prüft ob die Antwort vollständig ist"""
        full_response = "\n".join(response_lines + [line])
        end_markers = ["NICHTS GEFUNDEN", "etc...", "User:", "\x1b[91mUser:", "END_RESPONSE"]
        return any(marker in line or marker in full_response for marker in end_markers)
    
    def process_all_data(self, output_file: str = "results.txt"):
        """Hauptfunktion die alles verarbeitet"""
        try:
            # Dummy-Testdaten statt CSV
            data = [
                {'row_num': 1, 'land': 'Deutschland', 'textkennzeichen': 'TK1', 'gemeinde': 'Berlin', 'verwaltungssitz': 'Berlin'},
                {'row_num': 2, 'land': 'Frankreich', 'textkennzeichen': 'TK2', 'gemeinde': 'Paris', 'verwaltungssitz': 'Paris'}
            ]
            print(f"Verarbeite {len(data)} Test-Datensätze")
            
            self.start_gpt_process()
            results = []
            
            for row_data in data:
                print(f"\n{'='*60}")
                print(f"Verarbeite Zeile {row_data['row_num']}")
                print(f"Gemeinde: {row_data['gemeinde']}")
                print(f"Land: {row_data['land']}")
                print(f"{'='*60}")
                
                row_results = {
                    'row_num': row_data['row_num'],
                    'data': row_data,
                    'responses': []
                }
                
                for prompt_num, prompt_template in enumerate(self.prompts, 1):
                    # Teste verschiedene Methoden
                    if prompt_num == 1:
                        response = self.send_prompt(
                            prompt_template.format(**row_data),
                            row_data['row_num'],
                            prompt_num
                        )
                    elif prompt_num == 2:
                        response = self.send_prompt_alternative(
                            prompt_template.format(**row_data),
                            row_data['row_num'],
                            prompt_num
                        )
                    else:
                        response = self.send_prompt_encoded(
                            prompt_template.format(**row_data),
                            row_data['row_num'],
                            prompt_num
                        )
                    
                    row_results['responses'].append({
                        'prompt_num': prompt_num,
                        'response': response
                    })
                    time.sleep(2)  # Längere Pause zwischen Prompts
                
                results.append(row_results)
                self.save_results(results, output_file)
            
            print(f"\nAlle Ergebnisse gespeichert in: {output_file}")
            
        except Exception as e:
            print(f"Fehler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Beendet alle Prozesse sauber"""
        self.stop_reading = True
        
        # pexpect Prozess beenden falls vorhanden
        if hasattr(self, 'pexpect_child') and self.pexpect_child:
            try:
                self.pexpect_child.close(force=True)
                print("pexpect Prozess beendet")
            except:
                pass
        
        # subprocess Prozess beenden
        if self.process:
            try:
                self.process.stdin.close()  # Schließe stdin zuerst
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            print("GPT subprocess beendet")
    
    def save_results(self, results: List[Dict], output_file: str):
        """Speichert die Ergebnisse in eine Datei"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Ergebnisse der Test-Prompts - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            for result in results:
                f.write(f"ZEILE {result['row_num']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Gemeinde: {result['data']['gemeinde']}\n")
                f.write(f"Land: {result['data']['land']}\n")
                f.write(f"Textkennzeichen: {result['data']['textkennzeichen']}\n")
                f.write(f"Verwaltungssitz: {result['data']['verwaltungssitz']}\n\n")
                for response in result['responses']:
                    f.write(f"PROMPT {response['prompt_num']}\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"{response['response']}\n\n")
                f.write("=" * 80 + "\n\n")


def main():
    model_path = "gpt-oss-120b/original/"
    output_file = "email_search_results.txt"
    sender = GPTOSSPromptSender(model_path)
    try:
        sender.process_all_data(output_file)
    except KeyboardInterrupt:
        print("\nAbgebrochen durch Benutzer")
        sender.cleanup()
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
        sender.cleanup()


if __name__ == "__main__":
    main()
