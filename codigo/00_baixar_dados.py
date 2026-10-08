"""Download the original MovieLens 100k distribution; do not redistribute it."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

root = Path(__file__).resolve().parents[1] / "dados"
root.mkdir(exist_ok=True)
url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
with urlopen(url, timeout=60) as response:
    data = response.read()
with ZipFile(BytesIO(data)) as archive:
    for entry in archive.infolist():
        target = (root / entry.filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("Unsafe archive path")
    archive.extractall(root)
result = root.parent / "resultados"
result.mkdir(exist_ok=True)
record = {"url": url, "zip_sha256": hashlib.sha256(data).hexdigest(),
          "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted((root / "ml-100k").iterdir()) if p.is_file()}}
(result / "atividade_origem.json").write_text(
    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8",newline="\n")
print("MovieLens 100k obtido da distribuição original.")
print("SHA-256 do ZIP:", hashlib.sha256(data).hexdigest())
