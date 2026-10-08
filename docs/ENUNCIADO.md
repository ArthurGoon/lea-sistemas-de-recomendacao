# Atividade: recomendação de filmes

Fonte: [material de aula de Sistemas de Recomendação](https://teaching.vcoscrato.com/sisrec/), Victor Coscrato, Laboratório de Estatística Aplicada, 2026/2.

## Objetivo e requisitos

Comparar modelos de recomendação no MovieLens 100k, composto por 100.000 avaliações de 1 a 5, atribuídas por 943 usuários a 1.682 filmes. A tarefa solicita:

1. Uma análise exploratória breve da distribuição das notas e da quantidade de avaliações por usuário e por filme.
2. O treinamento de um modelo de fatorização matricial (SVD) e de um modelo User-Based KNN.
3. O cálculo do RMSE dos dois modelos por validação cruzada e de sua Precisão@10.
4. A comparação dos resultados e a discussão do modelo mais adequado ao problema.

## Correspondência com o relatório

| Requisito | Localização |
|---|---|
| Distribuição das notas e atividade por usuário e filme | Seção 1, Figura 1 e Tabela 1 |
| SVD e KNN de usuários | Seção 2.2 e Tabela 2 |
| RMSE com validação cruzada e Precisão@10 | Seções 2.1 e 3, Tabela 3 e Figura 2 |
| Comparação e discussão | Seções 3, 4 e 5, Figuras 2 a 4 |

Os modelos aleatório, por popularidade e por nota média funcionam como referências de desempenho. As ordenações do KNN por nota prevista e por soma ponderada são identificadas separadamente. Os diagnósticos de suporte e de ganhos por usuário aprofundam a comparação dos mesmos modelos, sem nova seleção orientada pelos testes externos.

O [relatório em PDF](../ATIVIDADE_Sistemas_de_Recomendacao_Arthur_Gon_RA811464.pdf) é o documento principal da entrega. Código, configurações e resultados neste repositório complementam sua leitura e permitem reproduzir a análise.
