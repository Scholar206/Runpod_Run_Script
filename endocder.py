
import chardet

with open("Data.csv", "rb") as f:
    result = chardet.detect(f.read(50000))  # ersten 50KB prüfen
print(result)
