"""Copy the final PDF and write a complete, nonduplicated SHA-256 manifest."""
from pathlib import Path
import hashlib
import shutil

ROOT=Path(__file__).resolve().parents[1]
PDF_SRC=ROOT/'relatorio/_build_ativ/atividade.pdf'
PDF_DST=ROOT/'ATIVIDADE_Sistemas_de_Recomendacao_Arthur_Gon_RA811464.pdf'
if not PDF_SRC.exists():
    raise SystemExit('Compile o relatório com python codigo/23_compilar.py.')
shutil.copy2(PDF_SRC,PDF_DST)
paths={PDF_DST,ROOT/'.gitattributes',ROOT/'.gitignore',ROOT/'README.md',ROOT/'requirements.txt',ROOT/'docs/ENUNCIADO.md'}
for pattern in ('codigo/*.py','relatorio/*.tex','relatorio/secoes/ativ_*.tex',
                'relatorio/tabelas/ativ_*.tex','resultados/atividade_*.json','resultados/atividade_*.csv','figuras/ativ_*.*'):
    paths.update(ROOT.glob(pattern))
# Record original data fingerprints without redistributing the files.
paths.update(p for p in (ROOT/'dados/ml-100k').glob('*') if p.is_file())
paths=sorted((p for p in paths if p.exists()),key=lambda p:p.relative_to(ROOT).as_posix())
lines=['MANIFESTO DA ATIVIDADE | Arthur Gon | RA 811464',
       'SHA-256 dos artefatos finais e dos dados originais locais','']
for path in paths:
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f'{digest}  {path.relative_to(ROOT).as_posix()}')
lines+=['',f'Total de arquivos: {len(paths)}']
(ROOT/'MANIFESTO_ATIVIDADE.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
print(f'PDF final compilado ({PDF_DST.stat().st_size//1024} KB). Manifesto: {len(paths)} arquivos.')
