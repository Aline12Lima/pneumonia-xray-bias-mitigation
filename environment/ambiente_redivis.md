# Ambiente de execução — Redivis

Registro do passo 2 da F2 (ambiente fixado e hardware). Medido em 24/10/2026 nos três workflows de auditoria: CheXpert Plus, RSNA e PadChest.

## Software

| Item | Versão |
|---|---|
| Python | 3.12.7 |
| PyTorch | 2.5.0+cu121 |
| torchvision | 0.20.0+cu121 |
| CUDA (do PyTorch) | 12.1 |
| cuDNN | 9.1.0 (90100) |
| Sistema | Linux 6.12.105, glibc 2.39 |

Os três notebooks têm a mesma imagem de software. A lista completa de pacotes (`pip freeze`, cerca de 299) está em `requirements_redivis_freeze.txt`, nesta pasta. Ela é o arquivo de lock do projeto.

## Hardware dos notebooks de auditoria

| Item | Valor |
|---|---|
| CPU | 16 núcleos |
| RAM | 125 GiB |
| Disco local (`/mnt/local/scratch`) | cerca de 560 GB livres |
| GPU | **nenhuma** (`torch.cuda.is_available() = False`) |

## Consequências

- **O treino precisa de uma configuração com GPU.** No Redivis, GPU é uma configuração de computação personalizada. Ela é paga por hora com créditos de computação. O notebook padrão gratuito não tem GPU (2 vCPUs, 32 GB, 6 h no máximo).
- **Os dados do CheXpert Plus não saem do Redivis.** Por isso, Colab e Kaggle não servem para M1 e M3: o treino com GPU acontece no próprio Redivis.
- **A decisão de GPU depende da medição do tempo por época** (`notebooks/bench_epoca.py`): primeiro em CPU, depois em GPU.
- **Sistema operacional:** o item "Windows ou WSL2" do plano não se aplica. Todo o processamento roda no Linux dos notebooks do Redivis; o computador local só edita e versiona.

## Tempo por época (`notebooks/bench_epoca.py`)

DenseNet121 com pesos ImageNet, 320 px, lote 16, 1.000 imagens sintéticas.

| Máquina | Treino | Inferência | Data |
|---|---|---|---|
| CPU, 16 threads, sem AMP | 166,7 s (6,0 img/s) | 39,5 s (25,3 img/s) | 24/10/2026 |
| GPU | a medir | a medir | — |

**Estimativa em CPU para os 21 treinos** (cerca de 15 épocas por treino; razão de negativos 1:4 no Plus): cerca de 11 h para o M1, 13 h para o M2 e 24 h para o M3. Somando a busca de taxa e as 5 sementes, dá cerca de 330 h, ou 14 dias de máquina ligada sem parar. Isso não cabe no prazo de 1 a 20/11 com folga. Conclusão: **o treino precisa de GPU**.

## Pendente

- Tempo de 1 época em GPU (mesmo script) e custo por hora da máquina.
- Escolha da máquina com GPU e orçamento em créditos para os 21 treinos.
