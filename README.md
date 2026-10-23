# Pneumonia X-Ray Bias Mitigation

Pesquisa sobre viés de luminosidade, robustez e generalização na classificação de pneumonia em radiografias de tórax.

## Organização

| Diretório | Conteúdo |
| --- | --- |
| `research/previous_study/` | Notebook, modelo e código do estudo anterior, com documentação histórica. |
| `data/raw/` | Originais em `chexpert_plus/`, `rsna/`, `padchest/` e `nih_metadata/` (somente CSV para mapear pacientes do RSNA). |
| `data/interim/` | Conversões para 8 bits, máscaras de ROI e features (F3). |
| `data/processed/` | Cache de imagens de 320 px com hash (5.1). |
| `data/splits/` | Listas de IDs e SHA-256 versionados, separados em `train/`, `val/` e `locked_test/` (4.8). |
| `notebooks/` | Diretório vazio; notebooks serão criados no VS Code/Jupyter em cada fase. |
| `src/` | Módulos futuros de dados, modelos, avaliação, perturbações, análise e visualização; `schemas/` reservado ao schema de predições (F2.6). |
| `experiments/` | Configurações YAML em `baseline/` (F5–F6), `generalization/` (F7), `perturbations/` (F8), `error_audit/` (F9, incluindo livro de códigos), `latent/` (F10) e `replication/` (F12). |
| `results/` | Figuras, tabelas, métricas, logs e `predictions/<run_id>/`, com `predictions.parquet` e `config.yaml` por execução. |
| `docs/` | Protocolo, fichas dos datasets, decisões, desvios, diário, literatura e reuniões. |
| `tests/` | Testes futuros, incluindo teste de fumaça (G2). |
| `research_notes/` | Notas de pesquisa organizadas por fase. |

## Estado atual

A nova estrutura está preparada; nenhum notebook da nova pesquisa foi criado.
As pastas antigas `train/`, `val/` e `test/` estão arquivadas em `research/previous_study/data/`.
O estudo anterior está em [research/previous_study/README.md](research/previous_study/README.md).
O arquivo `requirements.txt` permanece vazio. O `requirements.lock` foi criado com status pendente: o `.venv` local está incompleto, e o ambiente ainda precisa ser definido, validado e congelado (F2.2).

## Dados e artefatos

O conteúdo de `data/` é ignorado pelo Git, exceto `data/splits/`, que versiona listas de IDs e SHA-256. Modelos e resultados gerados também são ignorados. Arquivos `.gitkeep` preservam os diretórios versionáveis; as pastas de dados brutos devem ser recriadas em novos clones.
O modelo local foi movido para `research/previous_study/models/modelo_B_cbam_best.h5` e continua ignorado pelo Git.
Não há download automático de datasets ou modelos.

## Convenções da estrutura

- `docs/protocol/`: versões v1.0 e v1.1 congelada, em PDF com hash.
- `docs/dataset_notes/`: fichas Porta 1 e Porta 2 por base (F3).
- `docs/decisions/`: decisões em `D-0XX.md`.
- `docs/deviations/`: desvios em `DV-0XX.md`, somente após G4.
- `docs/journal/`: diário em `J-AAAAMMDD.md`.
- `data/splits/locked_test/`: reservado ao teste bloqueado; nenhum código de treino deve ler este diretório (G4). Não há código de treino novo implementado nem fiscalização automática dessa regra nesta estrutura.
- `results/predictions/<run_id>/`: criado a cada execução real; `<run_id>` é um marcador, não uma pasta literal.

Para recriar as pastas de dados ignoradas em um novo clone:

```bash
mkdir -p data/raw/{chexpert_plus,rsna,padchest,nih_metadata} data/interim data/processed
```
