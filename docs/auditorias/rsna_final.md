# Auditoria final — RSNA Pneumonia Detection Challenge (2018)

**Fase:** F3 — Datasets e auditoria
**Base:** RSNA Pneumonia (Stage 2), imagens do NIH ChestX-ray8 rotuladas por radiologistas
**Execução:** 18/10/2026 (Kaggle, notebook `audit-001-pneumonia`) e 21/10/2026 (Redivis, workflow `auditoria_rsna`, notebook `00_rsna_redivis`)
**Avaliadora da máscara:** Aline Lima

**Estado: base aprovada.** As quatro exigências para uma base entrar no estudo estão atendidas: positivos suficientes no teste, ID de paciente utilizável, máscara do pulmão aceitável em pelo menos 95 de 100 imagens, e termos de uso compatíveis.

---

## 1. Resumo

- **Base final:** 25.403 imagens de adultos (18 a 100 anos), **5.627 positivas (22,1%)**, do treino do Kaggle. Cerca de **1.125 positivos esperados no teste interno (20%)**, acima da meta de 750; o RSNA pode usar 3 faixas de qualidade.
- **Fonte oficial da RSNA**, com termos compatíveis. Os dados estão num dataset próprio no Redivis.
- **Rótulo de radiologistas, por imagem.** Positivo = "Lung Opacity"; negativo = "Normal" + "No Lung Opacity / Not Normal".
- **O pixel é uma imagem já processada duas vezes:** o NIH reduziu as radiografias para PNG de 8 bits, e a RSNA as regravou como DICOM com compressão JPEG com perda. Não há fabricante, janela nem parâmetros de exposição. **O RSNA fica fora da análise com parâmetros físicos.**
- **Maior risco de atalho: a projeção.** A prevalência é **37,9% no AP e 9,2% no PA**. No CheXpert Plus a relação é **inversa** (1,72% no AP e 3,19% no PA).
- **Máscara do pulmão aprovada** com o mesmo modelo e o mesmo procedimento das outras bases.

---

## 2. Fonte, dados e termos de uso

### 2.1 Duas fontes, mesmas imagens

| Fonte | Uso | Conteúdo |
|---|---|---|
| Kaggle — RSNA Pneumonia Detection Challenge, Stage 2 | Auditoria inicial (18/10) | 26.684 DICOM de treino com rótulo; 3.000 de teste sem rótulo |
| **RSNA, distribuição oficial (S3 público)** | **Fonte adotada (21/10)** | 30.000 DICOM; anotações adjudicadas de todas as etapas; mapeamento RSNA → NIH |

As duas fontes têm as mesmas imagens e rótulos. O treino da fonte oficial reproduz exatamente os 6.012 positivos do Kaggle.

### 2.2 Arquivos da fonte oficial

| Arquivo | Conteúdo | Tamanho |
|---|---|---|
| `pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip` | 30.000 DICOM | 3,98 GB |
| `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | Anotações do desafio (exportação do MD.ai) | 0,05 GB |
| `pneumonia-challenge-annotations-original_2018.json` | Anotações iniciais | 0,02 GB |
| `pneumonia-challenge-dataset-mappings_2018.json` | Mapeamento RSNA → NIH | 0,02 GB |

### 2.3 Termos de uso

- **Permitido:** pesquisa acadêmica, ensino e outros fins, comerciais ou não.
- **Proibido:** tentar identificar ou contatar pacientes.
- **Atribuição obrigatória:** Wang X et al., ChestX-ray8, CVPR 2017 (com link para o NIH e agradecimento ao NIH Clinical Center); Shih G et al., *Radiology: Artificial Intelligence* 2019, doi:10.1148/ryai.2019180041 (com link para a página da RSNA).
- **Dados derivados:** os termos não tratam do assunto; não há proibição.
- **Conclusão:** compatíveis. Usar a fonte oficial dispensa as regras da competição do Kaggle, que ficaram sem conferência.

### 2.4 Dataset no Redivis

**`rsna_pneumonia_2018` v1.0**, acesso restrito à autora, com descrição e atribuição registradas:

| Tabela | Conteúdo |
|---|---|
| `dicom` | 30.000 arquivos DICOM |
| `rotulos` | 30.000 linhas: mapeamento, paciente NIH, classe final, Exclude, marca do teste do Kaggle |
| `stage_2_sample_submission.csv` | 3.000 IDs do teste do Kaggle |
| `rsna_fontes` | Os 5 links oficiais |

Espaço ocupado: cerca de 4 GB.

---

## 3. Rótulos

### 3.1 Estrutura das anotações

O arquivo oficial guarda todas as rodadas de anotação do MD.ai em 7 grupos: as rodadas dos anotadores (com opacidade de alta, média ou baixa probabilidade), uma rodada de QA, a adjudicação (com "Exclude" e "Flag") e o grupo **"Calculated"**, que é o **rótulo final do desafio**.

| Classe final ("Calculated") | Imagens |
|---|---|
| Lung Opacity | 7.106 |
| No Lung Opacity / Not Normal | 12.788 |
| Normal | 9.790 |
| **Com rótulo final** | **29.684** |
| Sem classe final, marcada "Exclude" | 106 |
| Sem classe final, sem rótulo | 210 |

- Cada imagem rotulada tem uma única classe final, sem conflito.
- Positivos têm de 1 a 4 caixas de opacidade (média 1,6).
- O rótulo descreve **opacidade pulmonar sugestiva de pneumonia** vista pelo radiologista na imagem, não o diagnóstico clínico.

### 3.2 Treino × teste do Kaggle

| Conjunto | Lung Opacity | Not Normal | Normal | Total | Positivos |
|---|---|---|---|---|---|
| Treino | 6.012 | 11.821 | 8.851 | 26.684 | 22,5% |
| Teste | 1.094 | 967 | 939 | 3.000 | 36,5% |

- O teste foi **enriquecido de positivos** e, segundo a descrição do desafio, confirmado por três radiologistas: outro processo de seleção e de rotulagem.
- 1.575 pacientes aparecem nos dois conjuntos: o Kaggle separou por imagem. Não afeta o estudo, que divide por paciente.

---

## 4. Pixel e cabeçalho (origem da imagem)

Verificado nos 29.684 arquivos rotulados (cabeçalhos) e numa amostra de 200 imagens (pixels, 50 por combinação de projeção e rótulo, semente 2026).

| Item | Resultado |
|---|---|
| Tipo de DICOM | 100% *secondary capture* |
| Compressão | 100% JPEG Baseline **com perda** |
| Profundidade | 100% 8 bits, MONOCHROME2 |
| Tamanho | 100% 1024 × 1024 |
| Janela, fabricante, parâmetros físicos | Ausentes |
| `PixelSpacing` | 8 valores diferentes; provavelmente o espaçamento original do detector no NIH |
| `PatientID` do DICOM | Repete o ID da imagem; não serve como paciente |
| Texto gravado no pixel | Algumas imagens AP têm "PORTABLE" escrito na própria imagem |

**Conclusão:** a imagem passou por dois processamentos (PNG de 8 bits no NIH e JPEG com perda na RSNA). Brilho, contraste, nitidez e ruído medidos no RSNA descrevem **a imagem distribuída, já processada e comprimida**, não a radiografia como foi adquirida. O JPEG afeta especialmente nitidez e ruído, que só são comparados dentro de cada base, e se soma às degradações da perturbação controlada (eixo B).

---

## 5. Pacientes

- **Paciente = 8 primeiros dígitos do nome original no NIH** (por exemplo, `00012773_000.png` → paciente `00012773`; os 3 últimos dígitos são o número do exame de seguimento).
- Imagens rotuladas: 12.233 pacientes, mediana de 1 imagem por paciente, máximo de 75.
- Base final: 5.627 positivos vindos de 2.221 pacientes, cerca de 2,5 imagens positivas por paciente, provavelmente internados acompanhados com radiografias de leito.

---

## 6. Fluxograma da base final

| Etapa | Imagens | Pacientes | Positivos |
|---|---|---|---|
| 30.000 imagens do pacote oficial | 30.000 | 12.274 | — |
| Com rótulo final (sem "Exclude" e sem rótulo) | 29.684 | 12.233 | 7.106 |
| Só o treino do Kaggle | 26.684 | 11.452 | 6.012 |
| Idade de 18 a 100 anos | 25.403 | cerca de 10.926* | 5.627 |
| AP ou PA | 25.403 | cerca de 10.926* | 5.627 |
| Sem duplicata exata (MD5) | 25.403 | cerca de 10.926* | 5.627 |

\* Contagem feita antes de retirar as 5 imagens com idade acima de 100 (erros de registro do NIH, todas negativas); a diferença é de no máximo 5 pacientes.

Saíram: 316 sem rótulo final; 3.000 do teste do Kaggle (guardadas como conjunto de rótulo mais forte); 1.276 menores de 18 anos; 5 com idade acima de 100. Nenhuma duplicata exata.

---

## 7. Composição da base final

| Subgrupo | Imagens | Prevalência |
|---|---|---|
| AP | 11.459 | **37,9%** |
| PA | 13.949 | **9,2%** |
| Feminino | 11.021 | 21,3% |
| Masculino | 14.387 | 22,8% |
| 18–39 anos | 7.248 | 24,2% |
| 40–59 anos | 11.612 | 21,7% |
| 60–74 anos | 5.720 | 20,2% |
| 75+ anos | 828 | 22,9% |

(Tabela calculada com as 25.408 imagens antes de retirar as 5 com idade acima de 100.)

- **A projeção é o maior risco de atalho:** razão de chances de cerca de 6, e 77% dos positivos adultos são AP. O texto "PORTABLE" gravado no pixel reforça o risco.
- Sexo e idade alteram pouco a prevalência.

### Achado entre bases: a projeção aponta em direções opostas

| Base | Prevalência no AP | Prevalência no PA |
|---|---|---|
| RSNA | 37,9% | 9,2% |
| CheXpert Plus (rótulo da impressão) | 1,72% | 3,19% |

Um modelo treinado no RSNA pode aprender o atalho "imagem AP → pneumonia". No CheXpert Plus esse atalho aponta na direção errada, e o modelo tende a piorar justamente no teste externo. É uma explicação concreta e testável para a queda entre bases, e entra na análise de atalhos (T1) e na explicação da queda externa (eixo C).

---

## 8. Máscara do pulmão

### 8.1 Modelo e procedimento (os mesmos nas três bases)

- **Modelo:** `lungs-segmentation` resnet34, treinado só em Montgomery e Shenzhen, que nenhuma das três bases do estudo contém.
- **Procedimento:** entrada por percentis 0,5–99,5 (com inversão se MONOCHROME1; no RSNA todas são MONOCHROME2), redimensionada para 512 × 512, limiar 0,2, maior componente de cada pulmão, união dos dois.
- **Papel:** a máscara define onde medir; as medidas de qualidade são calculadas na imagem convertida pela regra oficial.

### 8.2 Inspeção

- **Amostra:** 50 AP e 50 PA entre os adultos do treino, semente 2026.
- **Às cegas:** as folhas mostravam só o número e a projeção, sem o rótulo.
- **Critério fixado antes de olhar:** falha se um pulmão está ausente ou com menos da metade coberta, se a máscara vaza claramente para fora do pulmão, ou se está no lugar errado. Duvidosa conta como falha.
- **Resultado (avaliação da Aline):** **100 de 100 aceitáveis.** Oito casos foram ampliados para análise detalhada (#5, #11, #15, #17, #58, #70, #85 e #91).

**Observações técnicas registradas:**

| Caso | Observação | Decisão da avaliadora |
|---|---|---|
| #11 (AP) | Parte do pulmão opacificado ficou fora da máscara | Aceitável: o pulmão não é visível na imagem |
| #70 (PA, paciente 13) | O contorno passa abaixo do diafragma e entra na bolha de ar do estômago | Aceitável: dentro da tolerância de borda |
| #85 (PA, paciente 7656) | Vazamento menor, também na bolha gástrica | Aceitável: dentro da tolerância de borda |

**Robustez:** mesmo contando esses três casos como falha, o resultado seria 97 de 100, acima do mínimo de 95. A aprovação não depende dessa discussão.

**Limitação:** o modelo marca o pulmão **com ar** e tende a deixar de fora o pulmão opacificado, o mesmo padrão visto no CheXpert Plus. Se a máscara encolher mais nos positivos, as medidas feitas dentro dela passam a refletir a doença. A comparação entre a área da máscara e o rótulo é obrigatória na passada completa, e a máscara central fixa fica como análise de sensibilidade.

**Teste da máscara nos positivos (26/10/2026):** 1.000 imagens elegíveis, 250 em cada célula projeção × rótulo, semente 2026, com a regra de decisão fixada antes de rodar (`src/imagem/teste_mascara_doenca.py`). Resultado: **a máscara não depende da doença** pela regra.

| Critério | Limite | AP | PA | Atingido? |
|---|---|---|---|---|
| 1. Cobertura das caixas no positivo ÷ mesma caixa num negativo pareado (mediana) | abaixo de 0,90 | 1,22 | 1,06 | Não (geral: 1,14) |
| 2. Área da máscara nos positivos em relação aos negativos | 10% menor, IC excluindo zero | −4,5% (IC −8,9% a +0,1%) | −4,7% (IC −8,2% a −1,0%) | Não |
| 3. Pulmão encolhido (razão abaixo de 0,35): positivos contra negativos | 2 vezes e 5 pontos a mais | 5,2% contra 1,6% | 1,2% contra 0,8% | Não |

Ressalvas para a leitura:

- **A referência transplantada é conservadora:** a caixa colocada no pulmão de outro paciente nem sempre cai no mesmo lugar da anatomia. Por isso a cobertura de referência (mediana 0,62) é menor que a dos positivos (0,74), e a razão passa de 1. O critério mostra que não há perda sistemática, não que a máscara cobre melhor a opacidade.
- **Existe uma cauda:** 17% dos positivos têm menos da metade da caixa dentro da máscara. São os casos muito opacos vistos nas inspeções (o pulmão doente vira uma ilha). A mediana não os captura.
- **A área é um pouco menor nos positivos** (cerca de 5%, com IC excluindo zero no PA). Fica abaixo da margem de 10%; pode ser inspiração menor nos doentes ou perda de borda.
- **Impacto nas medidas, descritivo:** preencher a máscara (envoltória convexa de cada pulmão) quase não muda a diferença de brilho entre positivos e negativos (variação de −0,01 desvio-padrão no AP e −0,01 no PA). No contraste, a variação é de −0,05 no AP e −0,14 no PA. A máscara central fixa continua como análise de sensibilidade.
- **Observação para a F4:** o brilho dentro do pulmão é maior nos positivos com as duas máscaras (diferença padronizada de 0,39 no AP e 0,16 no PA). Parte do "brilho" é a própria opacidade, e não a apresentação da imagem. Registrado como decisão aberta (F4 › Todas › Brilho e opacidade).

### 8.3 Máscara anterior (CheXmask)

Na auditoria de 18/10, a máscara do RSNA veio do **CheXmask** (máscaras prontas do modelo HybridGNet, treinado em JSRT, Montgomery, Shenzhen e uma parte do PadChest), com nota de qualidade de pelo menos 0,7 e 98 a 100 aceitáveis em 100. A avaliação mostrou falhas sistemáticas nas bases e nos cantos do pulmão, sem relação com o brilho (teste exato de Fisher, p = 1,0). **Foi substituída** pelo modelo único, porque o CheXmask viu imagens do PadChest no treino e não poderia ser usado nele.

---

## 9. Situação das exigências para a base entrar no estudo

| Exigência | Situação |
|---|---|
| Positivos suficientes no teste (750 com 3 faixas) | **Atende:** cerca de 1.125 |
| ID de paciente utilizável | **Atende:** pelo nome original no NIH |
| Máscara aceitável em pelo menos 95 de 100 | **Atende:** 100/100 (97/100 na leitura mais estrita) |
| Termos de uso compatíveis | **Atende:** fonte oficial da RSNA |

---

## 10. Decisões

Nomeadas por **Fase › Base › Tema**. A coluna "Referência antiga" traz o número usado nos chats anteriores, só para rastreio.

| Decisão | Conteúdo | Por quê | Referência antiga |
|---|---|---|---|
| **F3 › RSNA › Fonte** | Usar a distribuição oficial da RSNA, não o Kaggle | Termos compatíveis; mesmas imagens e rótulos | — |
| **F3 › RSNA › Rótulo** | Rótulo final do grupo "Calculated": positivo = Lung Opacity; negativo = Normal + Not Normal, com a classe guardada para sensibilidade | É o rótulo final do desafio, uma classe por imagem | — |
| **F3 › RSNA › Exclusões** | Excluir as 316 sem classe final, os menores de 18 anos e as idades acima de 100 | Fora do desafio; escopo de adultos; idades inválidas | — |
| **F3 › RSNA › Recorte** | Análise principal com o treino do Kaggle; as 3.000 do teste guardadas como conjunto de rótulo mais forte, para sensibilidade | Teste enriquecido (36,5%) e com outro processo de rotulagem | D-028 (emendada) |
| **F3 › RSNA › Paciente** | Paciente = 8 primeiros dígitos do nome original no NIH | Convenção do NIH; o `PatientID` do DICOM não serve | — |
| **F3 › RSNA › Física** | RSNA fora da análise com parâmetros físicos; medidas descritas como da "imagem distribuída, já processada e comprimida" | Sem fabricante, janela nem exposição; JPEG com perda | D-030 (parte) |
| **F3 › Todas › Contagem** | Positivos contados por imagem; intervalos de confiança por bootstrap de paciente; sensibilidade com cada paciente pesando o mesmo | O mesmo paciente pode ter a doença em fases diferentes; coerente com o PadChest | D-029 |
| **F3 › Todas › Máscara** | Mesmo modelo (`lungs-segmentation`) e mesmo procedimento nas três bases | Comparação justa entre bases; o CheXmask viu o PadChest | Substitui D-031 |

---

## 11. Pendências que não impedem a aprovação

| Item | Situação |
|---|---|
| Medidas de qualidade em todas as imagens e duplicatas por pixel | **Feito** na passada completa (seção 12) |
| Comparar a área da máscara com o rótulo | **Feito** no teste da máscara nos positivos (§8.2) |
| Verificar se o `PixelSpacing` pode substituir o fabricante | **Não pode**: os DICOM do RSNA não têm fabricante nem espaçamento do pixel (todos 1024 × 1024) |
| Recontar os positivos do teste depois da divisão por paciente (semente 2026) | **Feito**: 1.100 positivos no teste (seção 12) |
| Apagar o zip de 4 GB em `/out/rsna/` | Os arquivos já estão no dataset; libera espaço |

---

---

## 12. Passada completa e listas (28/10/2026)

**Passada** (`src/passada/passada_completa.py`, notebook gratuito, uma sessão, cerca de 0,5 a 1,2 s por imagem conforme a carga do servidor): as 25.408 imagens do arquivo de elegíveis abriram; cada uma teve as oito medidas na resolução original, a máscara, o MD5 dos pixels, o hash de diferença e o cache de 320 px em 8 bits (1,32 GB, mais 8,7 MB de tabelas em `/out/rsna/passada/`).

| Item | Resultado |
|---|---|
| Máscara vazia | 1 imagem (positiva, AP); fica no estudo, só as medidas dela ficam em branco |
| Razão entre pulmões abaixo de 0,35 | 393 (1,5%) |
| Pixels idênticos | 1 par, mesmo paciente (11723, negativo, AP): fica a imagem mais antiga (menor seguimento) |
| Quase idênticas (correlação ≥ 0,995 nas cópias de 320 px) | Só o mesmo par; entre pacientes diferentes, 4.371 pares abaixo de 0,95 e 1 entre 0,95 e 0,98 |

**Base final:** 25.402 imagens = 25.408 do arquivo de elegíveis − 5 com idade acima de 100 anos (o arquivo foi salvo antes do filtro) − 1 duplicata.

**Listas** (semente 2026, por paciente):

| Partição | Imagens | Pacientes | Positivos | Prevalência |
|---|---|---|---|---|
| Treino | 17.948 | 7.647 | 4.047 | 22,6% |
| Validação | 2.417 | 1.092 | 480 | 19,9% |
| Teste | 5.037 | 2.185 | 1.100 | 21,8% |

O teste tem 1.100 positivos, acima de 750: o RSNA usa 3 faixas de qualidade. Os SHA-256 estão em `data/splits/listas_sha256.json`.

## Referências

- Shih G et al. (2019). Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. *Radiology: Artificial Intelligence*. doi:10.1148/ryai.2019180041
- Wang X et al. (2017). ChestX-ray8: hospital-scale chest X-ray database and benchmarks. CVPR.
- RSNA Pneumonia Detection Challenge (2018): página oficial, termos de uso e atribuição. https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
- Modelo de máscara: https://github.com/alimbekovKZ/lungs_segmentation e https://github.com/alimbekovKZ/lungs_segmentation_train
- Gaggion N et al. CheXmask (máscara anterior). *Scientific Data*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11101488
