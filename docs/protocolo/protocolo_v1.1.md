# Protocolo científico v1.1

**Qualidade e apresentação de radiografias de tórax, erros, calibração e generalização de classificadores de pneumonia**

Autora: Aline Lima · Iniciação científica · Versão 1.1 (**congelada** em 31/10/2026) · substitui a v1.0 de 17/10/2026

Modelo fixo: DenseNet121 (pesos ImageNet) · Bases: CheXpert Plus, RSNA Pneumonia (Stage 2) e PadChest

> **Como ler esta versão.** A v1.1 é a v1.0 sem nenhum valor "a confirmar". Ela incorpora o que a auditoria dos dados (fase 3) mediu e as decisões do fechamento (fase 4). A partir do congelamento, qualquer mudança é um **desvio**, registrado com data, motivo e impacto em `docs/desvios/`. As decisões e a evidência de cada uma estão em `docs/decisoes.md`; as mudanças em relação à v1.0 estão resumidas na seção 14.

---

## 1. Resumo

Este protocolo investiga como a qualidade e a apresentação de radiografias de tórax se relacionam com os erros (falsos positivos e falsos negativos), a calibração e a generalização entre instituições de um classificador de pneumonia. O modelo é mantido fixo (DenseNet121 inicializada em ImageNet), para que as diferenças observadas venham dos dados e não da arquitetura. Três modelos (M1 a M3) são avaliados internamente e em fontes externas. Três eixos de análise (observacional, interventivo e de explicação da queda externa) e três testes de diagnóstico dos dados organizam a investigação, com duas análises exploratórias pré-especificadas (revisão visual dos erros e espaço latente). Nada neste documento é resultado do teste: o conjunto de teste está trancado por hash até 21/11/2026.

## 2. Problema e pergunta científica

### 2.1 Contexto e lacuna

- Modelos de pneumonia costumam ter desempenho externo menor que o interno: em 3 de 5 comparações, o desempenho em hospitais externos foi significativamente menor [1].
- O significado dos rótulos muda entre bases, o que é uma explicação alternativa à qualidade da imagem [3]; há diferenças de intensidade entre bases [2].
- Parâmetros de aquisição podem funcionar como atalho: a AUC de pneumotórax caiu de 0,94 para 0,56 quando as distribuições de exposição foram invertidas entre treino e teste [4].
- O efeito da qualidade pode ser pequeno: sob redução de dose simulada, a AUROC mudou menos de 0,005 em média, e a posição do paciente pesou mais que a dose [5].
- Testes de estresse com gama, contraste, brilho, nitidez e desfoque já existem [10].

**Lacuna candidata.** Numa busca estruturada de escopo (47 consultas; não é revisão sistemática), não apareceu estudo que combine: (i) qualidade natural medida; (ii) parâmetros físicos de aquisição; (iii) erros e calibração para pneumonia; (iv) teste em instituição independente; (v) a parcela da queda externa associada à qualidade. É uma lacuna candidata, não prova de ineditismo; a revisão sistemática formal continua pendente (seção 12).

### 2.2 Pergunta principal

Em fontes clínicas independentes, em que medida diferenças nas distribuições de características naturais de qualidade e de parâmetros de aquisição estão associadas à diferença de erro e de calibração de um classificador de pneumonia, após padronização de fatores clínicos, projeção e referência?

### 2.3 Hipóteses

- **H1.** Algumas características naturais de qualidade e apresentação estão associadas ao erro (FP e FN), após ajuste predefinido por posição, idade e sexo (e fabricante, quando existir).
- **H2.** A diferença de desempenho entre fontes diminui após a padronização dessas características, além do que a padronização clínica já reduz.
- **H3.** O efeito sobre as probabilidades (calibração) difere do efeito sobre a discriminação.
- **H4.** O resultado pode ser nulo ou heterogêneo. O protocolo não pressupõe que a qualidade seja a principal explicação. H4 é uma **regra de leitura**, não um teste.

### 2.4 Definições operacionais

| Termo | Definição |
|---|---|
| Qualidade natural | Propriedades dos exames como distribuídos, antes de qualquer perturbação experimental, medidas na região pulmonar (seção 6). "Natural" não significa sem processamento do fabricante |
| Apresentação | Posição (AP/PA), conversão para 8 bits e janelamento aplicados à imagem distribuída |
| Redação | "Qualidade e apresentação nas **imagens distribuídas**", nunca "na aquisição": nenhuma das três bases prova que a imagem não foi processada (auditoria, §4.3a) |
| Parâmetros físicos | Valores do cabeçalho (kVp, corrente, tempo, produto corrente-tempo, índices de exposição), com unidades e validade verificadas |
| Erro | FP e FN contra a referência de cada base, com limiar fixado na validação da fonte. Não é necessariamente erro clínico |
| Calibração | Correspondência entre probabilidades e frequências observadas (ECE, curva de confiabilidade, intercepto e inclinação) |
| Fonte externa independente | Outra origem clínica, sem pacientes ou imagens de teste no desenvolvimento |
| Parcela da queda | Redução padronizada do gap externo sob uma padronização explícita (seção 8.5). Não é fração causal |

### 2.5 Escopo

Dentro: pneumonia em adultos (18 a 100 anos), radiografias frontais, três bases, um modelo fixo, associação e padronização com incerteza. Fora (versões futuras): v2, aprendizado com laudo e imagem; v3, pediatria. Fora da v1: segunda arquitetura, afirmações causais, conclusões sobre o sinal cru. O conjunto **CheXthought** (anotações de radiologistas sobre imagens do Plus, divulgado em 28/10/2026) **não faz parte da v1.1**; se for usado depois do acesso, entra como análise adicional, registrada como desvio e rotulada como não pré-registrada.

## 3. Bases de dados e papéis

| Base | Origem | Papel | Pixels | Rótulo (análise principal) |
|---|---|---|---|---|
| CheXpert Plus | Stanford | Treino de M1 e M3; teste interno; externa de M2 | DICOM (Agfa, GE, 14 Varian) | CheXbert da **impressão** do laudo: positivo = 1; negativo explícito e não mencionado = 0; incerto excluído |
| RSNA Pneumonia (Stage 2) | NIH Clinical Center | Treino de M2 e M3; teste interno; externa de M1 | DICOM gerado a partir de PNG de 8 bits do NIH, com compressão JPEG com perda; sem fabricante, janela nem exposição | Grupo final do desafio: positivo = "Lung Opacity"; negativo = "Normal" + "Not Normal" |
| PadChest | Hospital Universitário de San Juan, Alicante | Só teste externo (M1, M2, M3) | PNG reescalados para 16 bits (Philips, Imaging Dynamics) | Laudos rotulados por médicos (pacotes 0–11 e 54) |

**Independência.** Três origens clínicas distintas (NIH, Stanford, Alicante); nenhum paciente em comum entre elas. O RSNA vem da distribuição oficial da RSNA (não do Kaggle); o paciente é identificado pelos 8 primeiros dígitos do nome original no NIH.

**Excluídas da v1:** CheXpert JPG (equalizado); NIH completo (contém o RSNA); MIMIC-CXR e VinDr-CXR (credenciamento); Kermany e PediCXR (pediatria).

**Termos de uso.** O Plus é só para pesquisa, sem uso comercial e sem redistribuição; o PadChest é sempre privado (cláusulas 5 a 7 do acordo). Publicam-se só código, contagens, hashes e resultados agregados; nenhuma imagem, ID ou medida por imagem sai do Redivis.

## 4. Auditoria dos dados (concluída)

A auditoria seguiu as duas portas da v1.0 (8 verificações de viabilidade e a ficha completa). Os resultados estão em `docs/auditorias/` e `docs/02_auditoria_dos_dados.md`. As três bases foram **aprovadas** pelos critérios da §4.4:

| Critério | Plus | RSNA | PadChest |
|---|---|---|---|
| Positivos no teste | 690 (≥ 500: **2 faixas** de qualidade) | 1.100 (≥ 750: 3 faixas) | 796 (≥ 750: 3 faixas) |
| ID de paciente utilizável | Sim | Sim (via NIH) | Sim |
| Máscara aceitável (inspeção de 100) | 95/100 | 100/100 | 97/100 |
| Termos compatíveis | Sim | Sim | Sim |

**Achados que mudam a análise:**

- **Proveniência do pixel (§4.3a).** Nenhuma base prova processamento mínimo; daí a redação "imagens distribuídas".
- **Parâmetros físicos (§4.3b).** No Plus, válidos em cerca de 20% das imagens, quase todas PA da GE; no RSNA, ausentes. O eixo A com parâmetros físicos é **rebaixado a exploratório**, sempre dentro do fabricante.
- **Rótulos (§4.3c).** No Plus, o arquivo do laudo inteiro está desalinhado com as imagens; o rótulo vem da impressão, com prevalência de cerca de 2,1%. Sensibilidades: incerto = 1; CheXbert dos achados.
- **Confundimento.** No Plus, projeção e fabricante andam juntos (AP é 99,3% Agfa; PA é 87,2% GE). No PadChest, fabricante, saturação e prevalência andam juntos. No RSNA, a prevalência é de 37,9% no AP e 9,2% no PA.
- **Máscara × doença.** Testada em 1.000 imagens do RSNA: a máscara **não depende da doença** pela regra fixada antes; mas o brilho dentro do pulmão é maior nos positivos (seção 8.3, regras de leitura).
- **Duplicatas.** 1 par de pixels idênticos no RSNA e 1 no Plus (mesmo paciente; sai a imagem mais recente); nenhuma duplicata quase idêntica entre pacientes diferentes nas três bases.

## 5. Divisão dos dados

**Regras.** Unidade de divisão: o paciente. Só frontais AP e PA; uma frontal por estudo (Plus: menor número de vista; PadChest: PA, depois AP, depois AP deitado; RSNA: já uma por estudo). Adultos de 18 a 100 anos; sexo registrado. Divisão 70/10/20 por paciente, **semente 2026**, estratificada pelo rótulo do paciente (positivo, só incerto, negativo). PadChest inteiro como teste externo. As medidas de qualidade não participam da divisão.

**Treino do Plus.** Todos os positivos do treino e **4 negativos por positivo**, sorteados **uma vez** (semente 2026); as 5 sementes de treino usam o mesmo conjunto e variam só a inicialização e a ordem dos lotes. Incertos fora do treino. Validação e teste com prevalência natural.

**RSNA.** Conjunto de treino do desafio (Stage 2) na análise principal; as 3.000 imagens do teste do Kaggle (enriquecido, outro processo de rotulagem) só como sensibilidade.

**M3.** União dos treinos de Plus e RSNA com amostragem balanceada por fonte; validação por fonte; limiar e temperatura na validação conjunta; teste externo no PadChest.

**Listas geradas (28 a 30/10/2026)**, `src/divisao/listas.py`. O SHA-256 é calculado sobre os IDs ordenados, um por linha; os IDs ficam só no Redivis.

| Base | Partição | Imagens | Pacientes | Positivos | SHA-256 (início) |
|---|---|---|---|---|---|
| Plus | Treino | 123.055 | 43.982 | 2.420 | `8886d430…` |
| Plus | Treino sorteado (usado pelo modelo) | 12.100 | 9.320 | 2.420 | `7f554299…` |
| Plus | Validação | 18.147 | 6.284 | 334 | `6e36fdc8…` |
| Plus | **Teste** | 35.457 | 12.566 | 690 | `7437a920…` |
| RSNA | Treino | 17.948 | 7.647 | 4.047 | `da15b3d1…` |
| RSNA | Validação | 2.417 | 1.092 | 480 | `eb13f9a0…` |
| RSNA | **Teste** | 5.037 | 2.185 | 1.100 | `c4de2e1c…` |
| PadChest | **Teste externo** | 24.770 | 22.985 | 796 | `0786ca66…` |

Os hashes completos estão em `data/splits/listas_sha256.json`. As listas de teste ficam numa pasta separada (`listas_teste_trancado/`) que nenhum código de treino lê (verificado por busca no repositório). **Nenhuma decisão é tomada olhando o teste.**

## 6. Pré-processamento e variáveis de qualidade

**Conversão para 8 bits** (regra única nas três bases): mínimo-máximo pelos percentis 0,5 e 99,5 da própria imagem, com inversão de MONOCHROME1; sem equalização nem normalização adaptativa. Redimensionamento direto para 320 × 320 (área). **Sensibilidade com a janela:** no PadChest, a janela do CSV reescalada, só no Philips (no Imaging Dynamics a janela corta as pontas e não serve como verificação); no Plus, a janela do DICOM como análise **exploratória descritiva** (concordância de brilho com os percentis de −0,04: a janela reordena as imagens sem cortá-las).

**Região de interesse.** Máscara pulmonar do `lungs-segmentation` (resnet34, treinado só em Montgomery e Shenzhen; nenhuma das três bases), mesmo procedimento nas três bases; máscara central fixa como sensibilidade.

**Medidas** (`src/qualidade/medidas.py`), na resolução original, sobre a imagem já convertida para 8 bits, dentro da máscara:

| Medida | Definição | Papel |
|---|---|---|
| Brilho | Média dos tons de 8 bits | Primária |
| Contraste | Desvio-padrão | Primária |
| Saturação | Fração dos pixels ≤ 1 ou ≥ 254 | Primária |
| Enquadramento | Fração da imagem em regiões pretas (≤ 5) ligadas à borda | Primária |
| Amplitude | p95 − p5 | Secundária |
| Entropia | Entropia (bits) do histograma de 256 tons | Secundária |
| Nitidez | Variância do Laplaciano 3 × 3 | Secundária, só dentro da base |
| Ruído | 1,4826 × MAD do resíduo (imagem − gaussiana σ = 1,5) nos 20% de pixels mais homogêneos | Secundária, só dentro da base |

Medidas auxiliares (área da máscara, saturação nativa) são só descritivas. As medidas já foram calculadas nas três bases: PadChest 24.773 imagens, RSNA 25.408, Plus 65.705 (as que o estudo usa).

**Parâmetros físicos:** só exploratórios, só dentro do fabricante, sem conversão em dose (Plus: GE; RSNA: ausentes; PadChest: campos do CSV dentro de Philips e Imaging Dynamics).

## 7. Modelagem

### 7.1 Por que DenseNet121 com pesos ImageNet

Arquitetura padrão em radiografia de tórax, o que favorece a comparação com a literatura; com poucos milhares de positivos por base, partir de pesos pré-treinados converge melhor que treinar do zero. O ImageNet não contém radiografias das bases usadas; pesos treinados em NIH ou CheXpert quebrariam a independência. As conclusões valem para esta arquitetura, inicialização e receita.

### 7.2 Receita (`configs/base.yaml`)

| Item | Valor |
|---|---|
| Entrada | 320 × 320, 8 bits, canal único replicado em 3, normalização do ImageNet |
| Aumento | Só geométrico: escala de 0,9 a 1,0 e rotação de até 5°. Nenhum aumento fotométrico |
| Perda | Entropia cruzada binária, com correção de prior pela subamostragem: logit corrigido = logit + ln(π_nat / (1 − π_nat)) − ln(π_treino / (1 − π_treino)), verificada na validação com prevalência natural |
| Otimização | AdamW, weight decay 1e-4, lote 32, cosseno, até 30 épocas, parada antecipada com paciência 5 pela AUPRC na validação da fonte |
| Busca | Taxa de aprendizado em {1e-4, 3e-4}, só na validação, mesma grade para M1 a M3 |
| Sementes | 5 (0 a 4) |
| Limiar | Sensibilidade de 0,90 na validação da fonte, reaplicado no externo (M3: validação conjunta) |
| Calibração | ECE bruto (principal) e ECE após temperatura ajustada na validação da fonte (secundário) |
| Recursos | GPU NVIDIA L4 no Redivis (cerca de 34 vezes mais rápida que a CPU no teste da fase 2); 21 treinos estimados em 10 a 20 h. Contingência (§11): 224 px ou 3 sementes, registrada como desvio |

### 7.3 Modelos

| Modelo | Treino | Teste interno | Teste externo |
|---|---|---|---|
| M1 | Plus | Plus | RSNA, PadChest |
| M2 | RSNA | RSNA | Plus, PadChest |
| M3 | Plus + RSNA (balanceado por fonte) | Plus e RSNA | PadChest |

São 9 pares modelo-base e 5 pares externos.

### 7.4 Registro e reprodutibilidade

Código, listas (por hash), ambiente e sementes versionados. Cada avaliação grava uma tabela de predições validada por `configs/schema_predicoes.yaml`: imagem, paciente, estudo, fonte, partição, modelo, semente, rótulo, probabilidade, posição, idade, sexo, fabricante, medidas de qualidade, degradação e nível. Todas as análises partem dessa tabela.

**Ensaio geral e trava do teste.** Antes de abrir o teste, todas as análises (E0, eixos A, B e C, T1 a T3) rodam sobre as partições de validação, para depurar o código; o código de análise é congelado com uma tag no git; só então o teste é destrancado (21/11/2026), com data, hora, hashes e tag registrados no diário.

## 8. Experimentos

### 8.1 E0: desempenho de referência e gap bruto

Para cada par modelo-base: AUROC, AUPRC (com a prevalência ao lado), FNR, FPR, ECE, intercepto e inclinação de calibração. Gap bruto por métrica. Sanidade: nenhum paciente em duas partições; treino com rótulos embaralhados deve dar AUROC perto de 0,5; sobreajuste proposital de um lote pequeno deve levar a perda a perto de zero.

### 8.2 Testes de diagnóstico dos dados

| Teste | Pergunta | Procedimento | Leitura |
|---|---|---|---|
| T1 | A qualidade carrega informação do rótulo? | Regressão logística e gradient boosting só com medidas de qualidade, posição, idade e sexo prevendo pneumonia, contra uma linha de base só com a posição; versão interna e cruzada. **Reportado com e sem o brilho** (seção 8.3) | AUROC perto de 0,5: pouca informação. Bem acima: risco de atalho, que pode ser clínico |
| T2 | Existe assinatura de origem? | Classificador de fonte (3 classes) só com medidas comparáveis entre bases, também dentro de positivos e de negativos | Acerto alto: comparar bases mistura origem com qualidade |
| T3 | O efeito tem dose-resposta? | Tendência monotônica no eixo B em pelo menos 2 dos 3 desfechos, controle negativo (p25/p75) sem efeito e mesma direção do eixo A | Convergência de A e B reforça H1; efeito abaixo da margem conta como nulo |

T1 e T2 rodam antes, em treino e validação, como **diagnóstico exploratório** (sem gastar o teste), e depois no teste, como análise confirmatória, com AUROC e intervalo de confiança, sem p-valor de hipótese.

### 8.3 Eixo A: observacional

Dentro de cada base, as medidas de qualidade são relacionadas a FN (entre positivos), FP (entre negativos) e ECE. Regressão logística com a qualidade padronizada em z por fonte, ajuste por posição, idade, sexo e fabricante (quando existir), e erro-padrão agrupado por paciente; efeito em razão de chances por 1 desvio-padrão. **As 5 sementes entram juntas na mesma regressão** (um p-valor por teste). ECE e inclinação de calibração por faixa de qualidade (Plus: 2 faixas; RSNA e PadChest: 3).

**Brilho e opacidade (regras fixadas antes do teste).** O brilho dentro do pulmão é maior nos positivos (diferença padronizada de 0,39 no AP e 0,16 no PA, no teste da máscara): parte do "brilho" é a própria opacidade. Por isso: (1) o brilho é lido principalmente no modelo de **falsos positivos** (entre negativos); no de falsos negativos, é reportado com a ressalva de que pode refletir a extensão da opacidade; (2) o T1 é reportado com e sem o brilho; (3) no eixo C, os pesos de S2 são estimados **dentro de cada classe**; (4) o eixo B é a verificação do efeito do brilho sem a doença no meio.

### 8.4 Eixo B: interventivo

**Onde cada degradação é aplicada.** Gama, contraste e brilho mudam cada pixel e quase não dependem da resolução: são aplicados na **imagem de 320 px**. Desfoque gaussiano, ruído gaussiano e redução de resolução (reduzir e reampliar) mudam a relação entre vizinhos: são aplicados na **resolução original**, depois da conversão para 8 bits e antes da redução para 320 px (`src/perturbations/degradacoes.py`).

**Níveis.** Cada degradação leva a sua medida aos percentis p25, p10, p5 e p1 (ou p75, p90, p95 e p99) do teste natural da própria base, na resolução em que é aplicada, para o lado em que age (desfoque e resolução: nitidez para baixo; ruído: para cima; gama e brilho: os dois lados; contraste: para baixo). O nível p25/p75 é o **controle negativo**. O **nível além da faixa** das degradações espaciais é o parâmetro que leva a medida, **medida em 320 px**, ao p5 (nitidez) ou p95 (ruído) natural em 320 px: um **controle positivo** de que a degradação chega ao modelo. Cada curva espacial reporta o efeito no original e em 320 px. O erro de ancoragem (medida após a degradação × percentil-alvo) é registrado. Os parâmetros finais de cada nível por base são fixados na fase 8, com a grade ampliada, antes de qualquer predição.

**Piloto (30/10/2026, 50 imagens da validação do Plus e do RSNA).** Brilho e contraste em 320 px concordam com o original (concordância de postos de 0,997 a 1,000). Nos níveis naturais, as degradações espaciais quase não chegam a 320 px (ruído: 0,01 a 0,1 desvio-padrão; desfoque: 0,1 a 0,35 no RSNA e menos no Plus), porque a redução para 320 px é de cerca de 6 vezes no Plus e 3 no RSNA. Daí o controle positivo.

**Amostra.** Todos os positivos do teste e 4 negativos por positivo, sorteados por paciente (semente 2026), a mesma para as 6 degradações e os 15 modelos (Plus: 3.450 imagens; RSNA: 5.037, o teste inteiro; PadChest: 3.980). AUROC, FNR e FPR não dependem da proporção de negativos; o ECE é calculado com pesos que devolvem a prevalência natural.

**Desfechos.** ΔAUROC, ΔFNR, ΔFPR e ΔECE em relação à imagem natural. **Teste de tendência** com todas as imagens: FNR e FPR por regressão logística do erro de cada imagem sobre o nível (0, 1, 2…), com erro-padrão agrupado por paciente; AUROC pela inclinação do ΔAUROC entre níveis, com bootstrap pareado por paciente. O Mann-Kendall sobre os pontos da curva é só descritivo (com 5 a 6 pontos, o menor p possível é cerca de 0,003).

### 8.5 Eixo C: explicação da queda externa

Para cada par fonte-externa: D_bruto = R_interno − R_externo, orientado para que positivo signifique piora no externo (para ECE e FNR, R_externo − R_interno).

- **S1:** padronização do externo para a distribuição clínica da fonte (posição, idade, sexo) por pesos de importância (classificador de domínio por regressão logística, pesos truncados). Gap D_S1.
- **S2:** S1 mais as medidas de qualidade comparáveis entre bases, com os pesos estimados **dentro de cada classe** (positivos e negativos separados). Gap D_S2.
- Parcela clínica = (D_bruto − D_S1) / D_bruto; parcela de qualidade = (D_S1 − D_S2) / D_bruto.

**Regras:** diferenças absolutas antes das razões; sem fração se D_bruto for próximo de zero; sem truncar valores negativos ou acima de 100%; reportar tamanho efetivo da amostra, diferença média padronizada após os pesos (meta < 0,1) e pesos extremos; reestimar os pesos em cada reamostragem do bootstrap.

**Evidências complementares:** suporte comum de qualidade; harmonização do externo por mapeamento de quantis do histograma da máscara; controles de rótulo ("Lung Opacity" do Plus contra o alvo do RSNA; subconjunto com baixa discordância entre rotuladores no Plus; subconjunto médico do PadChest, que é o recorte inteiro); ajuste de prior como sensibilidade. Formulação: "parcela do gap associada à padronização das características medidas". Decomposições formais [7, 8] ficam como análise secundária.

### 8.6 Análises exploratórias pré-especificadas

Exploratórias: com Benjamini-Hochberg quando houver p-valor, sem nenhuma decisão de modelagem baseada nelas.

**Revisão visual dos erros (fase 9).** Por par modelo × base externa: 25 FP e 25 FN de maior confiança e 25 FP e 25 FN sorteados (semente 0 do modelo), com IDs registrados por hash. Livro de códigos: dispositivos e tubos; marcadores e texto gravado na imagem (por exemplo "PORTABLE"); rotação ou corte; inspiração baixa; achado concorrente (derrame, atelectasia); rótulo provavelmente errado; exposição extrema. Codificação pela autora; se possível, um radiologista revisa 20% para concordância. Sem segundo avaliador, isso é declarado como limitação.

**Espaço latente e explicabilidade (fase 10, a primeira a ser cortada se faltar prazo).** Embeddings da penúltima camada (1.024 dimensões após o pooling) no teste, por modelo e semente. Sonda de origem: regressão logística linear prevendo a fonte a partir do embedding, com validação cruzada por paciente, comparada ao T2. Sonda de qualidade: regressão linear prevendo brilho e contraste (R²). Projeção 2D (UMAP) só descritiva. Saliência (Grad-CAM) numa amostra estratificada de VP, FP e FN: fração da saliência dentro da máscara pulmonar, entre tipos de erro e entre fontes, com teste de sanidade por randomização dos pesos [16]; se o mapa não muda, não é interpretado.

**Outras exploratórias:** janela do DICOM no Plus; parâmetros físicos dentro do fabricante; brilho no modelo de FN.

## 9. Métricas e análise estatística

| Dimensão | Métricas | Observação |
|---|---|---|
| Discriminação | AUROC (principal); AUPRC (secundária, com a prevalência ao lado) | A AUROC não depende da prevalência |
| Erros | FNR e FPR no limiar fixado; PPV como complemento | O PPV depende da prevalência |
| Calibração | ECE com 15 faixas de igual tamanho (principal); curva de confiabilidade; intercepto e inclinação; Brier como complemento | O ECE é enviesado com poucos positivos |
| Gap | D_bruto, D_S1, D_S2 e parcelas | Seção 8.5 |

**Incerteza.** Bootstrap por paciente (2.000 reamostragens) e dispersão entre as 5 sementes, reportados separadamente. Comparações pareadas por bootstrap pareado (DeLong para AUROC).

**Margens de relevância.** AUROC: 0,02. FNR e FPR: 0,03. ECE: o menor entre 0,02 e um quarto da prevalência da base (Plus cerca de 0,005; PadChest cerca de 0,008; RSNA 0,02), porque com prevalência de 2 a 3% o ECE fica perto de 0,01 e uma margem absoluta nunca seria alcançada. Precisão esperada (IC 95% de um AUROC isolado): ± 0,015 a 0,021.

**Leitura de cada comparação:** **relevante** se o intervalo de confiança inteiro fica acima da margem; **nulo na prática** se fica inteiro abaixo; **inconclusivo** se cruza a margem (alimenta o cenário S5).

**Multiplicidade: três famílias primárias, com Holm (α = 0,05) dentro de cada uma.**

| Família | Hipótese | Testes | Quantos |
|---|---|---|---|
| 1. Eixo A | H1 | Brilho, contraste, saturação e enquadramento × FN e FP × 9 pares modelo-base | 72 |
| 2. Eixo B | Resposta à piora da imagem | Tendência do ΔAUROC em gama, brilho e contraste × 9 pares | 27 |
| 3. Eixo C | H2 | Parcela de qualidade no AUROC nos 5 pares externos (M1 → RSNA e PadChest; M2 → Plus e PadChest; M3 → PadChest) | 5 |

Cada família corresponde a uma hipótese pré-registrada; uma família única de 104 testes tornaria impossível rejeitar os testes por bootstrap (com 2.000 reamostragens, o menor p é 0,0005). **Benjamini-Hochberg (secundárias):** medidas secundárias; degradações espaciais; FNR, FPR e ECE no eixo B; calibração por faixa; parâmetros físicos; parcela de qualidade nas outras métricas. **H3** é lida pelo padrão (os eixos B e C mexem mais no ECE que na AUROC?), sem teste próprio; **H4** é regra de leitura; **T1 e T2** são reportados com AUROC e intervalo, sem p-valor.

**Relato:** TRIPOD+AI, CLAIM, model card e um datasheet por base.

## 10. Resultados esperados e cenários

Expectativas a priori, não resultados: (1) queda externa e piora de calibração em pelo menos um par; (2) assinatura de origem detectável no T2; (3) associações modestas entre qualidade e erro, mais fortes nos extremos; (4) parcela de qualidade positiva, porém minoritária; (5) pelo piloto, curvas espaciais quase planas nos níveis naturais, com resposta só no controle positivo. Um resultado nulo é aceitável e publicável (H4).

| Cenário | Evidência | Interpretação |
|---|---|---|
| S1. Qualidade explica parte relevante | Parcela de qualidade com IC acima da margem e convergência A ↔ B (T3) | Qualidade medida associada a parte do gap. Não é prova causal |
| S2. Efeito pequeno ou heterogêneo | Efeitos abaixo da margem; controle negativo e tendência fracos | Coerente com [5]: a qualidade medida não é o principal fator |
| S3. Atalho dominante | T1 e T2 altos; efeitos grandes no eixo B | O classificador pode explorar origem ou exposição |
| S4. Gap não explicado | Parcela de qualidade perto de zero com gap persistente | Rótulo, população ou prevalência são hipóteses mais fortes [3] |
| S5. Inconclusivo | Intervalos que cruzam a margem; positivos insuficientes por estrato | Reportar a limitação de poder e as estimativas com IC |

Cada par é classificado num cenário **antes** de escrever a discussão. Critério de sucesso, em qualquer cenário: todas as análises pré-especificadas executadas, reprodutíveis, com incerteza e limitações declaradas.

## 11. Ameaças à validade e limitações

| Ameaça | Mitigação |
|---|---|
| Significado do rótulo muda entre bases; alvos diferentes (opacidade no RSNA, menção no laudo do Plus) | Controles de rótulo (§8.5); relatar como explicação alternativa |
| Ruído de rótulo no Plus | Referência por imagem no RSNA; subconjunto de baixa discordância entre rotuladores; sensibilidade incerto = 1 |
| Brilho no pulmão mistura apresentação e opacidade | Regras de leitura da §8.3; eixo B como verificação |
| Confundimento por projeção, fabricante e gravidade | Ajuste por posição; T1 contra linha de base só de posição; análises dentro do fabricante |
| Processamento dos pixels desconhecido; RSNA já convertido para 8 bits e comprimido | Redação "imagens distribuídas"; sensibilidade de conversão |
| Degradações espaciais quase apagadas pela redução para 320 px | Controle positivo e efeito em 320 px reportados (§8.4); conclusão restrita a esta resolução de entrada |
| Máscara mal segmentada | Inspeção por base; teste de dependência da doença; máscara central como sensibilidade |
| Poucos positivos no Plus (690) | 2 faixas de qualidade; leitura em três saídas com a margem |
| Prevalência diferente entre bases | AUROC e inclinação como principais; margem do ECE escalada; ajuste de prior como sensibilidade |
| Independência (RSNA derivado do NIH) | Papel restrito; PadChest como terceira origem |
| Múltiplas comparações | Três famílias com Holm; Benjamini-Hochberg nas secundárias |
| Arquitetura e inicialização únicas | Declarado como limitação |
| Recursos (GPU, cota de 10 GB no Redivis) | Créditos reservados à GPU; derivados no `/out`; contingência de 224 px ou 3 sementes |
| Uma só codificadora na revisão dos erros | Revisão de 20% por radiologista, se possível; senão, limitação declarada |

## 12. Cronograma, governança e pendências

| Fase | Período |
|---|---|
| Congelamento do protocolo (este documento) | 31/10 |
| F5 — Linha de base (M1 piloto, sanidade, determinismo) | 1 a 7/11 |
| F6 — M1 a M3 × 5 sementes, limiar, temperatura, ensaio geral | 8 a 20/11 |
| Geração das imagens degradadas da amostra (CPU, em paralelo ao treino) | 1 a 20/11 |
| **Destrancamento do teste** | **21/11** |
| F7 — Generalização (E0, T1, T2, eixo C) | 21/11 a 4/12 |
| F8 — Eixo B | 24/11 a 5/12 |
| F9 — Eixo A, revisão dos erros, T3 | 21/11 a 5/12 |
| F10 — Espaço latente (cortável) | 1 a 8/12 |
| F11 — Estatística e cenários | 6 a 12/12 |
| F12 — Reprodutibilidade | 8 a 15/12 |
| F13 — Redação e entrega | 13 a 31/12 (data final a confirmar) |

**Governança.** Marcos: auditoria concluída (30/10), protocolo congelado (31/10), teste destrancado só após o treino final e o congelamento do código de análise (21/11). Depois do congelamento, toda mudança é desvio (`docs/desvios/`), com data, motivo e impacto.

**Pendências que não mudam o que está congelado:** revisão sistemática formal (strings em PubMed, Scopus, Web of Science e Embase); T1 e T2 exploratórios em treino e validação; rótulos do rotulador CheXpert no Plus (para o controle de discordância); descrição de raça e etnia no Plus; acesso ao CheXthought (fora da v1.1).

## 13. Referências

1. Zech JR et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/
2. Pooch EHP, Ballester PL, Barros RC. Can we trust deep learning models diagnosis? The impact of domain shift in chest radiograph classification. https://arxiv.org/abs/1909.01940
3. Cohen JP et al. (2020). On the limits of cross-domain generalization in automated X-ray prediction. *MIDL/PMLR* 121. https://proceedings.mlr.press/v121/cohen20a.html
4. Shu HJ et al. (2026). Impact of exposure parameters on deep learning models in chest radiography and implications for deployment. *Radiology: Artificial Intelligence* 8(4), e250731. https://doi.org/10.1148/ryai.250731
5. Erenstein H et al. (2025). The effect of simulated dose reduction on the performance of artificial intelligence in chest radiography. *Journal of Imaging* 11(3), 90. https://doi.org/10.3390/jimaging11030090
6. Lotter W. (2024). Acquisition parameters influence AI recognition of race in chest x-rays. *Nature Communications* 15, 7465. https://doi.org/10.1038/s41467-024-52003-3
7. Zhang H et al. (2023). Why did the model fail? Attributing model performance changes to distribution shifts. *ICML/PMLR* 202. https://proceedings.mlr.press/v202/zhang23ai.html
8. Kobayashi Y et al. Underdiagnosis bias of chest radiograph diagnostic AI can be decomposed and mitigated via dataset bias attributions. *medRxiv*. https://www.medrxiv.org/content/10.1101/2024.12.16.24319063v2
9. Samei E et al. (2014). Automated characterization of perceptual quality of clinical chest radiographs. *Medical Physics* 41(11). https://doi.org/10.1118/1.4899183
10. Robustness stress testing in medical image classification. https://arxiv.org/pdf/2308.06889
11. Chambon P, Delbrouck JB et al. CheXpert Plus: augmenting a large chest X-ray dataset with text radiology reports, patient demographics and additional image formats. https://arxiv.org/pdf/2405.19538
12. RSNA Pneumonia Detection Challenge (2018). https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
13. Bustos A et al. (2020). PadChest: a large chest x-ray image dataset with multi-label annotated reports. https://bimcv.cipf.es/bimcv-projects/padchest/
14. DICOM PS3.3, módulo DX Image, C.8.11.3. https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.8.11.3.html
15. Shih G et al. (2019). Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.2019180041
16. Adebayo J et al. (2018). Sanity checks for saliency maps. *NeurIPS*. https://arxiv.org/abs/1810.03292

## 14. Mudanças da v1.0 para a v1.1

| Seção | v1.0 | v1.1 | Origem |
|---|---|---|---|
| §2.4, §11 | "Processamento mínimo" a verificar | Redação "imagens distribuídas" | Auditoria (F3 › Todas › Redação) |
| §2.5 | — | CheXthought fora da v1.1 | F4 › Plus › CheXthought |
| §3 | Rótulo do Plus por laudo; prevalência 2,7%; PadChest em recorte aleatório | Rótulo da impressão (≈ 2,1%), incerto excluído; PadChest = pacotes 0–11 e 54, rotulados por médicos; RSNA da distribuição oficial | Auditorias finais |
| §4 | Critérios a aplicar | Três bases aprovadas; Plus com 2 faixas | Auditorias finais |
| §5 | Razão de negativos, "primeira frontal" e recorte a definir | 4 negativos por positivo, sorteados uma vez; regra de primeira frontal por base; listas com SHA-256 | F4 › Todas › Receita; F4 › Todas › Listas |
| §6 | Conversão "janela quando existe, senão mínimo-máximo" | Percentis 0,5–99,5 em todas; janelas só como sensibilidade ou exploratória; definições exatas das 8 medidas; máscara `lungs-segmentation` | F3 › Todas › Conversão, Medidas, Máscara |
| §6, §8.3 | Eixo A com parâmetros físicos | Parâmetros físicos só exploratórios, dentro do fabricante | F3 › Plus › Física |
| §8.2, §8.3, §8.5 | — | Regras de leitura do brilho (FP como leitura principal; T1 com e sem brilho; pesos de S2 dentro da classe) | F4 › Todas › Brilho e opacidade |
| §8.4 | Resolução das degradações não definida; Mann-Kendall como teste | Fotométricas em 320 px, espaciais no original; controle positivo; amostra fixa; teste de tendência com todas as imagens | F4 › Todas › Degradações do eixo B; F4 › Todas › Testes primários |
| §8.6 | — | Revisão visual dos erros e espaço latente como exploratórias | Plano › Todas › Revisão dos erros; Espaço latente |
| §7.4 | — | Ensaio geral na validação e tag do código antes de abrir o teste | Plano › Todas › Ensaio geral |
| §9 | Margens propostas; família primária vaga | Margens confirmadas, ECE escalado pela prevalência; regra das três saídas; três famílias nominais (72, 27 e 5); sementes juntas na regressão | Plano › Todas › Estatística; F4 › Todas › Testes primários |
| §12 | Eixo A até 21/11 | Cronograma do plano de execução; pendências que não mudam o congelado | Plano de execução |

**Pré-registro.** O SHA-256 deste PDF é registrado no diário de 31/10/2026 e numa tag no git (`protocolo-v1.1`); se possível, também no OSF.
