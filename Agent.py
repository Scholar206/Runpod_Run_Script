#!/usr/bin/env python3
import csv
import requests
import json

# Konfiguration
INPUT_CSV = "data.csv"
OUTPUT_CSV = "email_results.csv"
GPT_SERVER_URL = "http://localhost:8000/v1/chat/completions"

def create_prompt(land, gemeinde, plz):
    return f"""E-MAIL & KONTAKTSUCHE
HAUPTAUFGABE
Finde direkte E-Mail-Adressen von zuständigen Personen/Behörden für Hochwasserschutz und Starkregenprävention am folgenden Standort:
Standortdaten:
Land: {land}
Gemeinde/Stadt: {gemeinde}
PLZ: {plz}
Suche zuerst nach spezialisierten Fachbereichen mit folgenden Suchbegriffen. Durchsuche dabei sehr gründlich die Website die du findest:
Tiefbauamt: {gemeinde} Tiefbauamt Stadtwerke Entwässerung Email
Wasserwehr: {gemeinde} Wasserwehr Feuerwehr Email 
Gib als Antwort nur dieses AUSGABEFORMAT sonst nichts.
email1@example.de
[Name Vorname] - [Funktion]
[Quelle-URL]
email2@example.de
[Name Vorname] - [Funktion]
[Quelle-URL]
email3@example.de
[Name Vorname] - [Funktion]
[Quelle-URL]
etc..."""

def send_to_gpt(prompt):
    payload = {
        "messages": [
            {"role": "system", "content": "Du bist ein Experte für Behördenkontakte. Nutze browsing um Websites gründlich zu durchsuchen."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "reasoning_effort": "high",
        "tools": [{"type": "browser"}]
    }
    
    response = requests.post(GPT_SERVER_URL, json=payload, headers={"Content-Type": "application/json"})
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"ERROR: {response.status_code}"

# CSV einlesen
with open(INPUT_CSV, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    next(reader)  # Header überspringen
    
    # Output CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Land', 'Gemeinde', 'PLZ', 'GPT_Response'])
        
        # Ab Zeile 6 starten (ersten 5 überspringen)
        for i, row in enumerate(reader):
            if i < 5:  # Zeilen 1-5 überspringen
                continue
            if i >= 10:  # Nach 5 verarbeiteten Zeilen (6-10) stoppen
                break
                
            land, gemeinde, plz = row[1], row[7], row[9]  # B=1, H=7, J=9 (0-basiert)
            
            print(f"Verarbeite: {land}, {gemeinde}, {plz}")
            
            prompt = create_prompt(land, gemeinde, plz)
            gpt_response = send_to_gpt(prompt)
            
            writer.writerow([land, gemeinde, plz, gpt_response])
            
print("Fertig! Ergebnisse in:", OUTPUT_CSV)
