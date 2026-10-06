# Pneumonia X-Ray Bias Mitigation

Pesquisa sobre viés de luminosidade, robustez e generalização na classificação de pneumonia em radiografias de tórax.

## Organização

| Diretório | Conteúdo |
| --- | --- |
| `research/previous_study/` | Notebook, modelo e código do estudo anterior, com documentação histórica. |
| `data/raw/` | Dados originais; contém as pastas preexistentes `train/`, `val/` e `test/`. |
| `data/interim/` | Dados intermediários. |
| `data/processed/` | Dados preparados para os experimentos. |
| `notebooks/` | Dez rascunhos numerados, do protocolo à reprodução final. |
| `src/` | Módulos futuros de dados, modelos, avaliação, perturbações, análise e visualização. |
| `experiments/` | Configurações de baseline, perturbações, generalização e replicação. |
| `results/` | Figuras, tabelas, métricas e logs gerados. |
| `docs/` | Protocolo, literatura, notas dos datasets e reuniões. |
| `tests/` | Testes futuros. |

## Estado atual

A nova estrutura está preparada; os novos notebooks são rascunhos sem execução ou resultados.
O estudo anterior está em [research/previous_study/README.md](research/previous_study/README.md).
O arquivo `requirements.txt` permanece vazio, como no projeto original: o ambiente de execução ainda precisa ser definido e validado.

## Dados e artefatos

Dados, modelos e resultados gerados não são versionados no Git. Arquivos `.gitkeep` preservam os diretórios da estrutura.
O modelo local foi movido para `research/previous_study/models/modelo_B_cbam_best.h5` e continua ignorado pelo Git.
Não há download automático de datasets ou modelos.
