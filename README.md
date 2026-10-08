# Sistemas de recomendação de filmes

**Arthur Gon · RA 811464**

Laboratório de Estatística Aplicada · 2026/2 · Docente: Victor Coscrato

## Relatório

**[Ler o relatório completo em PDF](ATIVIDADE_Sistemas_de_Recomendacao_Arthur_Gon_RA811464.pdf)**

O relatório de quatro páginas apresenta a análise exploratória do MovieLens 100k, a comparação entre SVD e KNN de usuários e a interpretação dos resultados. Este repositório reúne o código e os resultados que permitem examinar e reproduzir a análise.

A avaliação utiliza as cinco partições oficiais do conjunto de dados. Em cada partição, os hiperparâmetros são escolhidos em uma divisão interna do treino; o teste externo é reservado à avaliação. Os [requisitos da atividade](docs/ENUNCIADO.md) estão relacionados às respectivas seções do relatório.

## Resultados

Médias das cinco partições externas, com pesos iguais. Menores valores de RMSE e MAE e maiores valores de Precisão@10 e nDCG@10 indicam melhor desempenho.

| Modelo | RMSE | MAE | Precisão@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Aleatório | 1,6999 | 1,3902 | 0,0105 | 0,0113 |
| Mais populares | 1,0256 | 0,8203 | 0,1568 | 0,1982 |
| Melhores avaliados | 1,0937 | 0,8933 | 0,1179 | 0,1273 |
| KNN por nota | 0,9411 | 0,7346 | 0,0007 | 0,0007 |
| KNN por soma | 0,9411 | 0,7346 | **0,2014** | **0,2629** |
| SVD | **0,9098** | **0,7182** | 0,0753 | 0,0786 |
| Média global | 1,1256 | 0,9447 | — | — |

O SVD apresenta o menor erro de previsão de notas. O KNN por soma produz as listas com maior precisão neste protocolo, acrescentando, em média, 0,45 acerto por lista de dez em relação à popularidade. As duas ordenações do KNN usam o mesmo modelo ajustado: a primeira ordena pela nota prevista; a segunda, pela soma ponderada das contribuições dos vizinhos.

A Precisão@10 considera filmes com nota de teste ≥ 4 e usuários com ao menos um filme relevante no teste. Filmes já avaliados pelo usuário no treino são excluídos das recomendações. O relatório também examina a consistência entre partições, o suporte das recomendações e a distribuição dos ganhos entre usuários. As diferenças são descritivas, pois os treinos das cinco partições se sobrepõem.

## Material para consulta

| Material | Conteúdo |
|---|---|
| [Síntese dos resultados](resultados/atividade_sintese.json) | Médias e desvios-padrão, com precisão numérica completa |
| [Resultados por partição](resultados/atividade_rmse.json) | Métricas de cada modelo em cada teste externo |
| [Protocolo](resultados/atividade_protocolo.json) e [configurações](resultados/atividade_escolhas.json) | Dimensões das divisões, sementes verificáveis e parâmetros escolhidos |
| [Diagnósticos](resultados/atividade_diagnostico.json) e [resultados por usuário](resultados/atividade_usuarios.csv) | Suporte, acertos e comparações entre listas |
| [Verificação independente](resultados/atividade_verificacao.json) | Conferência de 120 métricas e das agregações dos diagnósticos |
| [Figuras](figuras/) e [fonte do relatório](relatorio/atividade.tex) | Visualizações e documento em LaTeX |
| [Código da análise](codigo/20_atividade.py) | Seleção interna e avaliação externa dos modelos |

## Reprodução

Requisitos: Python 3.12 e as bibliotecas de [requirements.txt](requirements.txt). A geração do PDF requer uma distribuição LaTeX com `pdflatex`, como TinyTeX, TeX Live ou MiKTeX.

Após obter uma cópia do repositório, executar os comandos a partir de sua pasta principal:

```bash
python -m venv .venv
```

Ativar o ambiente com `.venv\Scripts\Activate.ps1` no PowerShell ou `source .venv/bin/activate` no Linux/macOS. Em seguida:

```bash
python -m pip install -r requirements.txt
python codigo/00_baixar_dados.py
python codigo/testes.py
python codigo/20_atividade.py
python codigo/26_visoes_resultados.py
python codigo/25_verificar_resultados.py
python codigo/24_tabelas_relatorio.py
python codigo/23_compilar.py
```

O download obtém a distribuição original do GroupLens. Os 100.000 registros são preservados, sem exclusão ou imputação de notas. Posições sem avaliação permanecem ausentes; apenas no cálculo do cosseno seus desvios centrados são representados por zero. Médias e similaridades são calculadas exclusivamente com o treino de cada divisão.

A execução leva alguns minutos. Em cada treino externo de 80.000 avaliações, a divisão interna usa 64.000 para ajuste e 16.000 para seleção dos hiperparâmetros. O SVD utiliza 40 épocas seriais e semente 42; as divisões internas utilizam semente 42 mais o número da partição. O [ambiente de referência](resultados/atividade_ambiente.json) registra as versões utilizadas. Pequenas diferenças numéricas podem ocorrer entre plataformas.

A verificação inclui 44 testes matemáticos e a recomputação independente das métricas de teste. O compilador gera o PDF e o [manifesto de integridade](MANIFESTO_ATIVIDADE.txt), com SHA-256 dos arquivos e dos dados originais locais. `pdflatex` deve estar no PATH; alternativamente, seu diretório pode ser indicado pela variável `TEX_BIN`. No Windows, o compilador também procura o diretório padrão do TinyTeX.

## Dados e referências

O [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) é disponibilizado pelo GroupLens sob seus próprios termos. Os dados brutos são obtidos da fonte original pelo script de download; seus hashes estão em [atividade_origem.json](resultados/atividade_origem.json). As condições de uso acompanham a distribuição original.

Harper, F. M.; Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), artigo 19. [DOI: 10.1145/2827872](https://doi.org/10.1145/2827872).

Coscrato, V. (2026). [Sistemas de recomendação](https://teaching.vcoscrato.com/sisrec/). Material de aula, Laboratório de Estatística Aplicada, 2026/2.
