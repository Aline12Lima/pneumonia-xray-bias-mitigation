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
| GPU NVIDIA L4 (23 GB), g2-standard-4, lote 32, precisão mista | 4,9 s (205,7 img/s) | 1,6 s (624,1 img/s) | 25/10/2026 |

**Estimativa em CPU para os 21 treinos** (cerca de 15 épocas por treino; razão de negativos 1:4 no Plus): cerca de 11 h para o M1, 13 h para o M2 e 24 h para o M3. Somando a busca de taxa e as 5 sementes, dá cerca de 330 h, ou 14 dias de máquina ligada sem parar. Isso não cabe no prazo de 1 a 20/11. Conclusão: **o treino precisa de GPU**.

## Decisão de GPU (F2)

- **Máquina:** NVIDIA L4, g2-standard-4 (1 GPU, 4 vCPUs, 16 GB de RAM), no próprio Redivis. Os dados do CheXpert Plus não podem sair do Redivis.
- **Velocidade:** a L4 treina cerca de 34 vezes mais rápido que a CPU (205,7 contra 6,0 imagens por segundo).
- **Estimativa para os 21 treinos:** cerca de 10 h de GPU. Usando o dobro como margem, porque a leitura das imagens com só 4 vCPUs pode limitar a velocidade, o total fica entre 10 e 20 h.
- **Custo:** 4,53 créditos/h no Spot e 7,55 créditos/h no Standard. Cada crédito custa US$ 0,10. Total previsto: de 45 a 90 créditos no Spot, ou de 75 a 150 no Standard. A primeira compra foi de 100 créditos, em 25/10/2026.
- **Spot ou Standard:** fica para a F5. O piloto mede o tempo real de um treino. Como o checkpoint é salvo a cada época, dá para usar o Spot sem perder mais de uma época se a máquina cair.
- **Limite de 6 h por sessão:** cada treino leva bem menos que isso na L4 (o M3, o maior, deve ficar abaixo de 1 h).
- **Regra de uso:** a configuração com GPU só fica ligada enquanto roda; depois volta para a gratuita.

## Teste de fumaça (F2, passo 7)

`python -m src.fumaca`: 100 imagens fictícias passam por conversão, 320 px, máscara do pulmão, as oito medidas de qualidade (§6, `src/qualidade/medidas.py`), divisão por paciente, 1 época de DenseNet121, predição, correção de prior e tabela de predições conferida pelo schema.

| Máquina | Resultado | Data | Commit |
|---|---|---|---|
| CPU (notebook gratuito) | Passou: 8 de 8 conferências | 25/10/2026 | b10a58e |
| GPU NVIDIA L4 | Passou: 8 de 8 conferências | 25/10/2026 | b10a58e |
| CPU (notebook gratuito), com as oito medidas (F3, passo 3.12) | Passou: 9 de 9 conferências, incluindo `medidas_qualidade_completas`; máscara 46,6 s e medidas 0,65 s para 100 imagens | 25/10/2026 | 1c97467 |

A máscara ficou vazia em 100% das imagens fictícias. Isso é esperado: o modelo foi treinado em radiografias reais. A qualidade da máscara em imagens reais foi medida nas auditorias.

Pacotes instalados por cima da imagem do Redivis: `environment/requirements_extra.txt` (OpenCV) e `lungs-segmentation==1.1.2`, instalado com `--no-deps`.

## Mudanças de ambiente observadas

| Data | O que mudou | Efeito |
|---|---|---|
| 29/10/2026 | A imagem do Redivis trocou a versão do pandas: os avisos passaram a pedir numexpr 2.11.0 e bottleneck 1.5.0 (antes, numexpr 2.10.2) | Nenhum nas medidas (o pandas só monta as tabelas). A passada do Plus atravessou a troca; as sessões antes e depois usam o mesmo código de medida (`src/qualidade/medidas.py`, OpenCV e NumPy) |

## Passadas completas (F3)

| Base | Imagens | Sessões gratuitas | Ritmo |
|---|---|---|---|
| PadChest (segunda passada, 16 bits) | 24.773 | 2 | cerca de 1,4 s por imagem |
| RSNA | 25.408 | 1 | 0,5 a 1,2 s por imagem, conforme a carga do servidor compartilhado |
| Plus (recorte do estudo) | 65.705 | 4 | 0,8 a 2 s por imagem |

O ritmo varia com a ocupação do servidor compartilhado da configuração gratuita, não com o código. A sessão gratuita para em 6 h; a passada grava blocos de 200 imagens em `/out` e retoma de onde parou.
