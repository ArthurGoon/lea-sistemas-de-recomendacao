"""Portable LaTeX compilation, then package the final PDF and checksums."""
import os
from pathlib import Path
import runpy
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
tex = shutil.which("pdflatex")
if not tex and os.environ.get("TEX_BIN"):
    tex = str(Path(os.environ["TEX_BIN"]) / "pdflatex")
if not tex and os.name == "nt":
    candidate = Path(os.environ["APPDATA"]) / "TinyTeX/bin/windows/pdflatex.exe"
    if candidate.exists():
        tex = str(candidate)
if not tex:
    raise SystemExit("Instale uma distribuição LaTeX com pdflatex e adicione-a ao PATH (ou defina TEX_BIN).")
build = root / "relatorio/_build_ativ"
build.mkdir(exist_ok=True)
for iteration in range(1,4):
    proc = subprocess.run([tex,"-interaction=nonstopmode","-halt-on-error",
        "-output-directory=_build_ativ","atividade.tex"],cwd=root/"relatorio",capture_output=True)
    (build/f"pass{iteration}.log").write_bytes(proc.stdout+proc.stderr)
    if proc.returncode:
        raise SystemExit(f"Erro de compilação: consulte {build / f'pass{iteration}.log'}")
log = (build/"atividade.log").read_text(encoding="utf-8",errors="replace")
for problem in ("Overfull ","undefined references","Rerun to get cross-references right"):
    if problem in log:
        raise SystemExit(f"Verifique a diagramação/referências: {problem}")
runpy.run_path(str(root/"codigo/21_pacote_atividade.py"),run_name="__main__")
