# Auditoria final — CheXpert Plus (F3)

**Fase:** F3 · **Passos tocados:** 3.2, 3.3, 3.4, 3.8, 3.9, 3.10 (parte 1), 3.11, §4.3e
**Datas de execução:** 2026-10-19, 2026-10-20 e 2026-10-21
**Ambiente:** Redivis, workflow `auditoria_chexpert_plus`, notebook `00_plus_auditoria` (Python; pydicom 3.0.2; lungs-segmentation 1.1.2 instalado sem dependências; opencv-python-headless)
**Versão do documento:** v2 (2026-10-21). Substitui a v1, que usava o rótulo do laudo inteiro, depois descoberto desalinhado (§4.2.3).

**Estado:** **Portão G3 do Plus aprovado com ressalvas.** Os quatro critérios do G3 estão atendidos, sendo o de positivos pela regra alternativa de 2 faixas. Features, duplicatas por pixel e rótulos do rotulador CheXpert ficam para a passada completa e não bloqueiam o G3 (§5, D-037).

---

## 1. Resumo

- **Base elegível após todos os filtros propostos:** 176.660 imagens frontais AP/PA, 62.832 pacientes, **3.444 positivos** pelo CheXbert da impressão.
- **Prevalência:** 1,95% com incerto contado como negativo; 2,13% com incerto excluído. O valor de 2,7% do protocolo deixa de valer.
- **Positivos esperados no teste interno (20%):** cerca de 689. Abaixo de 750 e acima de 500, então vale a regra do protocolo de **2 faixas de qualidade** (cerca de 345 por faixa, acima do mínimo de 250).
- **O arquivo de rótulos do laudo inteiro (`report_fixed.json`) está desalinhado**: seus rótulos não correspondem ao texto da imagem da mesma linha. Os arquivos de impressão e de achados estão alinhados.
- **ROI pulmonar aprovada:** 95 de 100 imagens aceitáveis na inspeção, exatamente no limite do G3.
- **Parâmetros físicos em cerca de 20% da base**, quase todos em PA da GE. O eixo A físico deve ser rebaixado a exploratório.
- **Projeção, fabricante, presença de janela e prevalência andam juntos:** o AP é quase todo Agfa, sem janela, com prevalência de 1,72%; o PA é majoritariamente GE, com janela, com prevalência de 3,19%.
- **A proveniência do pixel não é provável pelo cabeçalho.** A redação usa "qualidade e apresentação nas imagens distribuídas".

---

## 2. Fonte, ambiente e arquivos gerados

### 2.1 Tabelas do dataset CheXpert Plus v1.0 no Redivis

| Tabela | Referência qualificada | Linhas / arquivos | Tamanho | Uso |
|---|---|---|---|---|
| `df_chexpert_plus_240401` | `df_chexpert_plus_240401:bavj` | 223.462 linhas, 27 colunas | — | Metadados, laudos, demografia |
| `DICOM_train` | `dicom_train:1934` | 210.740 | 2.578,7 GB | Imagens e cabeçalhos |
| `DICOM_valid` | `dicom_valid:mg73` | 229 | 2,9 GB | Imagens e cabeçalhos |
| `PNG_train` | `png_train:s6cj` | 223.228 | 719,3 GB | Não usado (D-007) |
| `PNG_valid` | `png_valid:41v9` | 234 | 0,8 GB | Não usado |
| `CheXpert Labels` | `chexpert_labels:pmec` | 3 arquivos JSONL | 0,3 GB | Rótulos do CheXbert (§4.2) |
| `DICOM_compressed` | — | 18 pacotes `.zip` | cerca de 2,74 TB (estimado) | Não usado (custo) |
| `PNG_compressed` | — | não inspecionada | — | Não usado |
| `RadGraph XL Annotations` | — | não inspecionada | — | Fora da v1 (D-017) |

Os 15 pacotes `.zip` com tamanho lido têm cerca de 155,7 GB cada; o chunk 17 tem 90,6 GB. O total supõe tamanho semelhante para os chunks 4, 7 e 10.

### 2.2 Arquivos persistidos em `/out` do notebook

| Arquivo | Conteúdo |
|---|---|
| `fluxograma_plus_v2.csv` | **Vigente.** Fluxograma com o rótulo da impressão |
| `plus_elegiveis_v2.csv` | **Vigente.** 176.660 elegíveis com rótulo da impressão |
| `fluxograma_plus.csv`, `plus_elegiveis.csv` | v1, com o rótulo do laudo inteiro (desalinhado). Mantidos só como histórico |
| `cabecalhos_teste16.csv`, `cabecalhos_amostra2000.csv` | Cabeçalhos do teste de vazão e da amostra do 3.3/3.4 |
| `roi_teste3.png`, `roi_teste3_comparacao.png`, `roi_teste3_final.png` | Desenvolvimento do procedimento de ROI |
| `roi_inspecao_1.png` a `roi_inspecao_4.png`, `roi_inspecao_100.csv` | Inspeção das 100 imagens |
| `roi_inspecao_100_ficha.csv`, `roi_casos_discussao.png`, `roi_casos_5_41_51.png` | Ficha técnica das 100 e casos ampliados |

### 2.3 Notas técnicas do ambiente

- `redivis.file()` foi descontinuado; a forma atual é `redivis.table("referência").file("nome_do_arquivo")`.
- Com dados de exportação restrita, o notebook não acessa a internet depois de iniciado. Pacotes e pesos entram por **Dependencies**, e só são instalados ao parar e iniciar o notebook (reiniciar o kernel não basta).
- O `lungs-segmentation` 1.1.2 exige versões de 2020 (`torch==1.7.1`, `numpy==1.17.4` e outras) que não instalam no Python atual. Solução: `pip install --no-deps` no `post_install.sh`, seguido da criação do modelo para baixar os pesos durante a instalação.
- A função de inferência do pacote chama a GPU obrigatoriamente; a inferência foi reescrita para CPU com o mesmo pré-processamento.
- "Error displaying widget: model not found" vem das barras de progresso de downloads em paralelo e é inofensivo. O aviso do `numexpr` também.

---

## 3. Porta 1 — oito verificações (§4.2)

| # | Verificação | Resposta | Evidência | Documentado ou verificado |
|---|---|---|---|---|
| 1 | Alvo e quem rotulou | Pneumonia mencionada na **impressão** do laudo; CheXbert, por imagem | Arquivo de impressão alinhado com o texto (§4.2.3) | Verificado |
| 2 | ID de paciente utilizável | Sim: `deid_patient_id`, 64.725 pacientes | Tabela principal | Verificado |
| 3 | Vistas frontais filtráveis | Sim: `frontal_lateral` e `ap_pa` | 191.071 frontais; 17 com LL ou RL | Verificado |
| 4 | Positivos após filtros | 3.444; cerca de 689 no teste | Fluxograma v2 (§4.4) | Verificado (estimativa de teste) |
| 5 | Imagens abrem e batem com a tabela | PNG: 223.462 = tabela. DICOM: 210.969 casam, 0 órfãos, 12.493 sem DICOM. Cerca de 2.230 DICOM lidos sem falha, 106 com pixels abertos | Blocos 2, 9b a 19 | Verificado em amostra |
| 6 | Formato, bits e resolução | DICOM; BitsAllocated 16; BitsStored 12, 14 ou 15; exemplos de 2021 × 2021 a 2800 × 3408, com AP frequentemente mais largo que alto | Amostra de cabeçalhos | Verificado na amostra; resumo completo na passada completa |
| 7 | Sobreposição com outra base | Origem única (Stanford). Duplicatas exatas internas tratadas | MD5 (§4.3) | Verificado (exatas); perceptuais pendentes |
| 8 | Termos de uso | Compatíveis: só pesquisa, sem uso comercial, sem redistribuição. Guardar só derivados | Artigo do CheXpert Plus, §4.5 | Documentado |

---

## 4. Porta 2 — ficha por eixo

### 4.1 Proveniência e tamanho (3.8, pendência 2)

- DICOM soltos somam **2,58 TB** (cerca de 12 MB por arquivo). Não cabem no disco do notebook (60 GB).
- **Recorte de pacientes inviável:** a base inteira rende só cerca de 689 positivos de teste.
- **Vazão:** 31,2 s para o primeiro arquivo isolado (latência); **54,4 MB/s com 8 downloads simultâneos**, cerca de 0,2 s por arquivo. A amostra de 2.000 cabeçalhos levou 1,9 min; 100 imagens com segmentação, 43 s.
- **Estimativa da passada completa:** cerca de 10 h de download e cerca de 24 h de segmentação em CPU (0,4–0,5 s por imagem).

### 4.2 Rótulos e anotações (3.9, §4.3c, D-018)

#### 4.2.1 Origem das contagens do protocolo

Os números do protocolo (6.042 / 18.771 / 2.806 / 195.843) são os da Tabela 3 do artigo do CheXpert Plus: rotulador CheXpert, por imagem. Esse rotulador não está disponível na cópia do Redivis.

#### 4.2.2 Rótulos disponíveis

A tabela `CheXpert Labels` traz `report_fixed.json`, `impression_fixed.json` e `findings_fixed.json`, com os mesmos nomes da pasta `chexbert_labels/` do repositório oficial. **São rótulos do CheXbert.**

| Fonte (223.462 imagens) | Positivo | Incerto | Negativo | Não mencionado |
|---|---|---|---|---|
| Protocolo (CheXpert, Tabela 3) | 6.042 | 18.771 | 2.806 | 195.843 |
| CheXbert, laudo inteiro (**desalinhado**) | 5.963 | 23.045 | 4.720 | 189.734 |
| CheXbert, impressão | 4.847 | 19.367 | 3.322 | 195.926 |
| CheXbert, achados | 915 | 3.702 | 381 | 218.464 |

Desempenho dos rotuladores segundo o artigo (F1 macro nos 14 achados, não específico de pneumonia): CheXbert da impressão 0,34 contra a imagem e 0,93 contra o texto; CheXbert dos achados 0,44 e 0,65. O F1 de 0,44 citado no protocolo é o dos achados.

#### 4.2.3 Desalinhamento do arquivo do laudo inteiro

**Primeiro sinal.** Na base elegível, os positivos de pneumonia pelo laudo inteiro (4.684) e pela impressão (3.444) coincidem em só **77 imagens**. Se os dois rótulos fossem independentes, o esperado seria cerca de 91. A concordância nas quatro categorias foi 76,0%, igual à esperada pelo acaso.

**Teste de alinhamento.** O CheXbert só marca um achado (1, 0 ou −1) quando o texto fala dele. Comparou-se a fração de imagens com o achado marcado quando o texto da seção correspondente menciona o termo e quando não menciona:

| Arquivo × texto | Achado | Marcado quando menciona | Marcado quando não menciona |
|---|---|---|---|
| report × laudo | Pneumonia | 15,3% | 15,0% |
| report × laudo | Derrame | 65,2% | 64,5% |
| report × laudo | Dispositivos | 61,4% | 58,4% |
| impression × impressão | Pneumonia | 99,9% | 7,4% |
| impression × impressão | Derrame | 99,4% | 2,9% |
| impression × impressão | Dispositivos | 95,1% | 29,3% |
| findings × achados | Pneumonia | 99,7% | 1,2% |
| findings × achados | Derrame | 99,6% | 0,7% |
| findings × achados | Dispositivos | 93,9% | 3,2% |

**Conclusão:** o `report_fixed.json` não corresponde às imagens das suas linhas e foi descartado. Os arquivos de impressão e de achados estão alinhados. É um problema da cópia distribuída e merece aviso aos mantenedores.

**Consequências observadas antes da correção:** a prevalência "plana" entre projeções, sexos, idades e fabricantes, e o caso #41 da inspeção (pneumonia incerta e opacidade positiva pelo laudo inteiro, com impressão descrevendo pulmões limpos), eram efeitos do arquivo desalinhado.

#### 4.2.4 Rótulos no conjunto elegível final (impressão)

Positivo 3.444; incerto 15.291; negativo 2.003; não mencionado 155.922.

### 4.3 Integridade técnica

**Cobertura do DICOM.** 12.493 imagens (5,6%) não têm DICOM; nenhum DICOM sem linha na tabela.

| Projeção (todas as imagens) | Imagens | Sem DICOM |
|---|---|---|
| AP | 161.622 | 9.628 (5,96%) |
| PA | 29.432 | 1.353 (4,60%) |
| Lateral | 32.391 | 1.512 (4,67%) |

Falta de DICOM por rótulo da impressão (frontais AP/PA): positivo 5,52%; não mencionado 5,75%; incerto 5,95%; negativo 4,27%. **Os positivos não faltam mais que os demais**, então a exclusão não enviesa a prevalência.

O artigo afirma 223.228 imagens disponíveis em DICOM. A diferença para os DICOM soltos (cerca de 155 GB) é compatível com o tamanho das imagens faltantes (cerca de 153 GB). Hipótese não verificada: os pacotes `.zip` contêm a base completa.

**Duplicatas exatas (MD5) no conjunto elegível:** 6 grupos, 12 arquivos.

| Tipo | Grupos | Imagens | Observação |
|---|---|---|---|
| Entre pacientes diferentes | 3 | 6 | Idades diferentes; num grupo, sexo diferente |
| Mesmo paciente, estudos consecutivos | 3 | 6 | Estudos 1–2, 8–9 e 17–18 |

### 4.4 Fluxograma de exclusões (rótulo da impressão)

| Etapa | Imagens | Pacientes | Positivos | Saíram |
|---|---|---|---|---|
| Total | 223.462 | 64.725 | 4.847 | — |
| Frontais | 191.071 | 64.710 | 3.731 | 32.391 laterais |
| AP ou PA | 191.054 | 64.710 | 3.730 | 17 com projeção LL ou RL |
| Com DICOM | 180.073 | 62.933 | 3.524 | 10.981 sem DICOM |
| Idade ≥ 18 e registrada | 179.853 | 62.833 | 3.522 | 220 (3 menores, 217 sem idade) |
| 1 frontal por estudo | 176.670 | 62.833 | 3.444 | 3.183 frontais repetidas |
| Sem duplicata exata | 176.661 | 62.833 | 3.444 | 9 |
| Sexo registrado | 176.660 | 62.832 | 3.444 | 1 com sexo "Unknown" |

### 4.5 Pixels: proveniência (3.3, §4.3a)

- Cabeçalho com **22 elementos** no arquivo examinado em detalhe (o artigo cita até 47 no conjunto).
- **Ausentes:** `ImageType`, `PresentationIntentType`, `LossyImageCompression`, `PixelIntensityRelationship`, `SOPClassUID`.
- **Presentes no exemplo Agfa:** `MONOCHROME1`, `PresentationLUTShape = INVERSE`, `RescaleType = P-VALUES`, `RescaleSlope = 1`, `RescaleIntercept = 0`, `BurnedInAnnotation = NO`.
- `RescaleType = P-VALUES` sugere valores de apresentação, ou seja, imagem já processada. É indício, não prova.
- O artigo garante pixels idênticos aos do CheXpert 1.0, e o JPG público foi gerado com mínimo-máximo e equalização de histograma, o que confirma a exclusão do JPG.

**Conclusão:** proveniência não provável pelo cabeçalho; redação "qualidade e apresentação nas imagens distribuídas".

### 4.6 Pixels: parâmetros físicos (3.4, §4.3b, pendência 1)

**Amostra:** 1.000 AP e 1.000 PA do conjunto elegível, semente 2026. Denominador: 2.000 cabeçalhos. Estratificação por projeção, porque o fabricante só é conhecido após a leitura.

| Fabricante (normalizado) | AP | PA | Total | MONOCHROME1 | Com janela | BitsStored |
|---|---|---|---|---|---|---|
| AGFA | 642 | 44 | 686 | 100% | 8% | 12, 15 |
| AGFA-GEVAERT AG | 351 | 84 | 435 | 100% | 1% | 12 |
| GE HEALTHCARE | 7 | 248 | 255 | 0% | 100% | 14 |
| GE MEDICAL SYSTEMS | 0 | 624 | 624 | 0% | 100% | 14 |

Por família: AP é 99,3% Agfa; PA é 87,2% GE. Ponderando pela projeção, a base é **cerca de 86% Agfa e 14% GE**.

**Fração com valor válido (> 0):**

| Fabricante | kVp | Corrente | Tempo | mAs | µAs | EI | EI-alvo |
|---|---|---|---|---|---|---|---|
| AGFA | 0,12 | 0,12 | 0,00 | 0,12 | 0,12 | 0,12 | 0,12 |
| AGFA-GEVAERT AG | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| GE HEALTHCARE | 1,00 | 1,00 | 0,00 | 1,00 | 1,00 | 0,15 | 0,15 |
| GE MEDICAL SYSTEMS | 1,00 | 1,00 | 0,00 | 1,00 | 1,00 | 0,00 | 0,00 |

**Mediana (mínimo–máximo) entre válidos:**

| Fabricante | kVp | Corrente (mA) | mAs | µAs | EI | EI-alvo |
|---|---|---|---|---|---|---|
| AGFA | 90 (85–120) | 400 (320–500) | 2 (2–10) | 2.400 (1.600–10.000) | 385 (162–1.216) | 400 (400–564) |
| GE HEALTHCARE | 120 (90–130) | 392 (100–576) | 3 (1–10) | 2.637 (1.009–10.403) | 156,2 (88,4–241,9) | 192,3 (115,0–194,6) |
| GE MEDICAL SYSTEMS | 110 (100–130) | 400 (160–500) | 2 (1–17) | 2.245 (880–17.400) | — | — |

**Fabricante × rótulo da impressão (proporção de positivos na amostra):** AGFA 1,6%; AGFA-GEVAERT AG 1,6%; GE HEALTHCARE 2,4%; GE MEDICAL SYSTEMS 2,7%.

**Conclusões:**

- Zeros em kVp e exposição foram tratados como ausentes.
- Tempo de exposição ausente em todos; derivável como mAs ÷ mA.
- `ExposureInuAs` é a versão precisa de `Exposure` (mAs, inteiro).
- Índices de exposição com escalas diferentes por fabricante, o que confirma a D-013.
- **Cerca de 20% da base tem parâmetros físicos válidos**, quase todos em PA da GE.
- **O fabricante se associa ao rótulo** na mesma direção da projeção (GE e PA com mais positivos).

### 4.7 Composição (§4.3e), rótulo da impressão

| Subgrupo | Imagens | Prevalência (incerto = 0) | Prevalência (incerto excluído) |
|---|---|---|---|
| AP | 149.341 | 1,72% | 1,89% |
| PA | 27.319 | 3,19% | 3,46% |
| Feminino | 73.794 | 1,90% | 2,09% |
| Masculino | 102.866 | 1,98% | 2,17% |
| 18–39 anos | 24.525 | 2,12% | 2,28% |
| 40–59 anos | 53.878 | 1,67% | 1,81% |
| 60–74 anos | 53.942 | 1,76% | 1,92% |
| 75+ anos | 44.315 | 2,43% | 2,72% |

- **A prevalência no PA é quase o dobro da do AP** (3,19% contra 1,72%). É o **oposto** do confundimento por indicação que o protocolo previa (AP de leito com pacientes mais graves). Uma explicação possível é que exames de seguimento à beira do leito descrevam "opacidades estáveis" em vez de "pneumonia". Como projeção e fabricante andam juntos, **o T1 ganha peso**: posição e fabricante, sozinhos, carregam informação do rótulo.
- **A idade tem padrão em U:** prevalência menor entre 40 e 74 anos e maior nos extremos.
- **Sexo não altera a prevalência.**
- **O AP cresce com a idade:** 80,2% entre 18 e 39 anos; 89,8% a partir de 75.
- **Imagens por paciente:** mediana 1, média 2,8, máximo 91.

### 4.8 Dependência

- **Entre origens:** o Plus é a única base de Stanford; não há sobreposição com NIH (RSNA) ou Alicante (PadChest).
- **Dentro da base:** duplicatas exatas tratadas (D-028). Duplicatas por pixel pendentes para a passada completa.

### 4.9 Região de interesse pulmonar (3.11, pendência 4)

#### 4.9.1 Escolha do modelo

| Modelo | Dados de treino | Situação |
|---|---|---|
| TorchXRayVision PSPNet | ChestX-Det, subconjunto do NIH ChestX-ray14 | Descartado: poderia ter visto imagens do RSNA |
| **`lungs-segmentation` resnet34** | Montgomery e Shenzhen (NLM), via Kaggle "chest-xray-masks-and-labels"; Dice 0,966 na validação do autor | **Escolhido** |
| lungVAE (Selvan et al., 2020) | Montgomery e Shenzhen; licença MIT | Reserva |

Nenhum dos dados de treino do modelo escolhido vem do NIH, de Stanford ou de Alicante.

#### 4.9.2 Procedimento congelado (desenvolvido em 3 imagens, semente 7)

1. **Entrada do segmentador:** normalização por percentis 0,5–99,5 do DICOM, com inversão de MONOCHROME1, redimensionada para 512 × 512, limiar 0,2.
2. **Pós-processamento:** manter só o maior componente conectado de cada pulmão; ROI = união dos dois.
3. **Separação de papéis:** a máscara define onde medir; as features são medidas na imagem convertida pela regra oficial (D-030).

**Desenvolvimento:** com mínimo-máximo puro, uma PA da GE ficou "lavada" (pixels extremos comprimem o tórax em cinza médio) e a máscara cobriu só parte dos pulmões (área 0,12). Com percentis, os pulmões passaram a ser cobertos (área 0,21), mas surgiram ilhas fora do tórax (abdome, bolha gástrica). O maior componente por pulmão removeu as ilhas sem alterar os contornos.

#### 4.9.3 Inspeção de 100 imagens

- **Amostra:** 50 AP e 50 PA, semente 2026, sem as 3 imagens de desenvolvimento. A metade PA serve como teste de estresse.
- **Critério fixado antes de olhar:** falha se um pulmão está ausente ou com menos da metade coberta, se a máscara invade claramente fora do pulmão, ou se está no lugar errado. Casos duvidosos contam como falha.
- **Avaliadora:** Aline Lima, com conferência dos casos limítrofes em figura ampliada.
- **Resultado: 95 aceitáveis em 100 — passa no critério do G3 (≥ 95), exatamente no limite.**

| Falha | Projeção | Fabricante | Tipo | Contexto do laudo |
|---|---|---|---|---|
| #5 | AP | Agfa | Vazamento lateral | Tubo endotraqueal, linhas, opacidades bibasais, derrame pequeno à esquerda |
| #9 | AP | — | Vazamento lateral | — |
| #41 | AP | Agfa | Encolhimento: um pulmão reduzido a um círculo no ápice | Avaliação limitada por biotipo e posição; rotação; pulmões limpos |
| #51 | PA | GE | Vazamento para gás fora do pulmão | Dreno com gás subcutâneo; cólon interposto sobre o fígado |
| #56 | PA | — | Vazamento | — |

**Ficha das 100 (descritiva; 5 falhas não permitem testar associação):** as falhas se distribuem por igual entre projeções (3/50 AP, 2/50 PA), fabricantes (3/56 Agfa, 2/44 GE), bits, dispositivos, opacidade, derrame e edema.

**Alerta automático:**

- Limites de área total (< 0,10 ou > 0,45) e de área por pulmão (< 0,03) marcaram só o #23, que foi julgado aceitável, e não pegaram nenhuma falha.
- A **razão entre as áreas dos dois pulmões** pega o encolhimento: o #41 tem a menor razão das 100 (0,31). Um corte em 0,35 marcaria o #41 e mais duas aceitáveis (#98 e #26).
- **Vazamentos não são detectáveis por área** (razão de 0,75 a 0,99). A proteção contra eles é a sensibilidade com ROI central fixa (§6 do protocolo).

**Risco registrado:** o modelo tende a segmentar pulmão com ar, não pulmão anatômico. Se a ROI encolher mais nos positivos, as features medidas na ROI passam a refletir a doença. A checagem ROI × rótulo é obrigatória na passada completa.

---

## 5. Pendências do Plus

| Item | Passo | Bloqueia G3? | Onde será feito |
|---|---|---|---|
| Oito features na ROI, resolução original | 3.12 | Não | **Feito** nas 65.705 imagens que o estudo usa (§7) |
| Duplicatas por pixel (hash perceptual) | 3.10 | Não | **Feito** (§7) |
| Resumo de resolução e abertura de pixels | 3.2 (itens 5 e 6) | Não | **Feito** nas 65.705 imagens (§7) |
| Alerta de ROI pela razão entre pulmões; checagem área da ROI × rótulo | 3.11 | Não | **Feito**: razão em §7; área × rótulo no teste da máscara (`rsna_final.md` §8.2) |
| Rótulos do rotulador CheXpert (CSV do CheXpert 1.0) | §4.3c, §8.5 | Não | Download do CSV, sem imagens. Agora é o único segundo rotulador para o controle de discordância |
| T1 e T2 exploratórios em treino e validação | 3.13 | Não | Listas prontas (§7); falta rodar |
| Aviso aos mantenedores sobre o `report_fixed.json` | — | Não | Opcional |
| Registrar formalmente as decisões abaixo | F4 | Sim, para G4 | `docs/decisoes/` |

---

## 6. Conclusão — decisões

A auditoria de tabelas, rótulos, cabeçalhos e ROI do Plus está concluída e o **G3 do Plus está aprovado com ressalvas**. As decisões abaixo estão **propostas** e viram "tomadas" na F4, antes do congelamento (G4). Os IDs a partir de D-024 são provisórios.

| ID | Decisão proposta | Evidência | Seção do protocolo |
|---|---|---|---|
| D-018 (revisada) | Rótulo primário: **CheXbert da impressão**. Positivo = 1; negativo explícito e não mencionado = 0; **incerto excluído**. Sensibilidades: incerto = 1; CheXbert dos achados. **O arquivo do laudo inteiro é descartado** | Laudo inteiro desalinhado (§4.2.3); impressão alinhada; F1 contra o texto 0,93 | §4.3c |
| D-024 | **Ambiente:** notebooks do Redivis, processamento em fluxo (baixar, ler cabeçalho, segmentar, calcular features, gravar cache 320 px, apagar o original), 8 downloads simultâneos, inferência em CPU | 54,4 MB/s; disco de 60 GB | §7.4, §11 (F2) |
| D-025 | **Excluir as imagens sem DICOM** (10.981 frontais AP/PA). Não usar os pacotes `.zip` | Positivos faltam 5,52% contra 5,75% dos não mencionados | §5 |
| D-026 | **Adulto = idade ≥ 18 e registrada** | 3 menores e 217 sem idade | §2.5, §5 |
| D-027 | **Primeira frontal = menor número de vista dentro do estudo** | Remove 3.183 | §5 (A7) |
| D-028 | **Duplicatas exatas:** excluir todos os membros de grupos entre pacientes; dentro do mesmo paciente, manter o estudo mais antigo | 9 imagens | §4.3d, §5 |
| D-029 | **Excluir sexo não registrado** | 1 imagem | §5 |
| D-030 (revisada) | **Conversão do Plus: mínimo-máximo por percentis 0,5–99,5 para todas as imagens**, com inversão de MONOCHROME1. Janela da GE e mínimo-máximo puro como sensibilidades | Janela só na GE; mínimo-máximo puro "lava" imagens com pixels extremos (§4.9.2) | §6 (altera) |
| D-031 | **Eixo A físico exploratório**, restrito à GE e dentro do fabricante. Eixo A primário com features de imagem | Parâmetros válidos em cerca de 20% da base | §4.4, §8.3 |
| D-032 | **Fabricante em duas famílias (Agfa, GE).** Ajuste por posição; declarar que posição e fabricante não se separam no Plus | AP 99,3% Agfa; PA 87,2% GE | §8.3, §8.5, §11 |
| D-033 | **Sem recorte de pacientes no Plus** | Cerca de 689 positivos de teste com a base inteira | §5 |
| D-034 | **Atualizar o texto do protocolo:** contagens e prevalência passam a sair do CheXbert da impressão após filtros (1,95% / 2,13%); o F1 de 0,44 é do CheXbert-achados; o confundimento previsto (AP mais grave) não aparece no rótulo | §4.2, §4.7 | §3, §11 |
| D-035 | **Confirmar a redação "imagens distribuídas"** | Elementos de proveniência ausentes | §4.3a |
| D-036 | **ROI:** `lungs-segmentation` resnet34 (Montgomery + Shenzhen), entrada por percentis 0,5–99,5, 512 px, limiar 0,2, maior componente por pulmão, união. ROI central fixa como sensibilidade | Inspeção: 95/100 | §6 (pendência 4) |
| D-037 | **Features (3.12), duplicatas por pixel e resumo de resolução feitos na mesma passada do cache da F5 (5.1)** | As duas tarefas baixam as mesmas 176.660 imagens; cerca de 34 h de máquina | Plano F3/F5 |
| D-038 | **Plus usa 2 faixas de qualidade** nas análises estratificadas do teste interno | Cerca de 689 positivos de teste (regra de 500 do G3) | §4.4, §8.3 |

**Estado do portão G3 para o Plus:**

| Critério | Situação |
|---|---|
| Positivos no teste | Atende pela regra de 2 faixas (cerca de 689 ≥ 500) |
| ID de paciente utilizável | Atende |
| ROI aceitável em pelo menos 95 de 100 | Atende (95/100, no limite) |
| Termos de uso compatíveis | Atende |

**Ressalvas:** positivos abaixo da meta de 3 faixas; ROI no limite e sem detector de vazamento; rótulo do laudo inteiro inutilizável; projeção, fabricante e prevalência confundidos.

---

---

## 7. Passada completa e listas (28 a 30/10/2026)

**Recorte da passada** (F4 › Plus › Passada): só as imagens que o estudo usa, 65.705 = treino sorteado (12.100) + validação (18.148) + teste (35.457). Quatro sessões do notebook gratuito, cerca de 0,8 a 1 s por imagem; cache de 320 px com 3,57 GB e tabelas com 25,8 MB em `/out/plus/passada/`.

| Item | Resultado |
|---|---|
| Abriram | 65.705 de 65.705 |
| Máscara vazia | 4 imagens |
| Razão entre pulmões abaixo de 0,35 | 1.125 (1,7%) |
| Pixels idênticos | 1 par, mesmo paciente (patient41910, estudos 3 e 4, negativo, validação): fica o estudo 3 |
| Quase idênticas (correlação ≥ 0,995) | Só o mesmo par; entre pacientes diferentes, 6.636 pares, todos abaixo de 0,95 |

**Fabricante e resolução** (nome unificado; mediana):

| Fabricante | Imagens | Linhas × colunas | Espaçamento |
|---|---|---|---|
| Agfa (AGFA, Agfa, Agfa-Gevaert AG) | 56.575 | 2320 × 2828 (2048 × 2494 em "AGFA") | 0,15 a 0,17 mm |
| GE ("GE Healthcare", GE Healthcare, GE MEDICAL SYSTEMS) | 9.116 | cerca de 2021 × 2021 | não gravado |
| Varian | 14 | 3052 × 3052 | 0,139 mm |

**Percentis × janela gravada no DICOM** (11.478 imagens com janela): concordância de postos baixa, brilho −0,04, contraste 0,55, amplitude 0,53, entropia 0,48, nitidez 0,61, ruído 0,68. Com a janela, o brilho mediano cai de 152 para 103, sem cortar a imagem (saturação mediana zero nas duas). Decisão: F3 › Plus › Conversão (percentis como principal; janela como análise exploratória descritiva).

**Listas** (semente 2026, por paciente; base de 176.659 imagens depois da duplicata):

| Partição | Imagens | Pacientes | Positivos | Incertos | Prevalência (incerto excluído) |
|---|---|---|---|---|---|
| Treino | 123.055 | 43.982 | 2.420 | 10.678 | 2,15% |
| Treino sorteado (4 negativos por positivo) | 12.100 | 9.320 | 2.420 | 0 | 20% |
| Validação | 18.147 | 6.284 | 334 | 1.548 | 2,01% |
| Teste | 35.457 | 12.566 | 690 | 3.065 | 2,13% |

O teste tem 690 positivos, acima de 500: vale a regra de 2 faixas (cerca de 345 por faixa). Retirar a duplicata mudou só o hash da validação; treino, treino sorteado e teste ficaram idênticos. Os SHA-256 estão em `data/splits/listas_sha256.json`.

## Referências usadas nesta auditoria

- Chambon P, Delbrouck JB et al. *CheXpert Plus: augmenting a large chest X-ray dataset with text radiology reports, patient demographics and additional image formats.* arXiv:2405.19538. Tabela 3, Tabela 4, §4.3, Apêndices A e B.
- Repositório oficial: https://github.com/Stanford-AIMI/chexpert-plus (estrutura `chexbert_labels/`).
- Modelo de ROI: https://github.com/alimbekovKZ/lungs_segmentation (pacote e pesos) e https://github.com/alimbekovKZ/lungs_segmentation_train (dados de treino).
- Reserva de ROI: Selvan R et al. (2020), https://github.com/raghavian/lungVAE
- Documentação do Redivis, notebooks e acesso à internet: https://docs.redivis.com/reference/projects/notebook-nodes/notebook-concepts
