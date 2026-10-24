# 01 — Problema, pergunta e hipótese

**Projeto:** Qualidade e apresentação de radiografias de tórax e erros, calibração e generalização de classificadores de pneumonia
**Etapa do ciclo:** 1 de N (definição do problema)
**Base:** Protocolo científico v1.0 (17/10/2026), com os ajustes que a auditoria dos dados (etapa 02) tornou necessários
**Versão:** 23/10/2026 · Aline Lima
**Situação:** proposta para o congelamento do protocolo (31/10/2026). Até lá, mudanças aqui são decisões; depois, viram desvios registrados.

---

## Resumo em cinco linhas

| Item | Resposta |
|---|---|
| Pergunta | Em fontes clínicas independentes, quanto da diferença de erro e de calibração de um classificador de pneumonia está associado a diferenças na qualidade e na apresentação das radiografias, depois de padronizar fatores clínicos, projeção e referência? |
| Decisão que informa | Na validação externa de um classificador, a qualidade da imagem precisa ser medida e padronizada, ou a queda se explica por outros fatores (população, rótulo, prevalência)? |
| Público | Pesquisadores e desenvolvedores de IA em radiologia; equipes de física médica e de qualidade de imagem; quem avalia levar um modelo de uma instituição para outra |
| Unidade de análise | A radiografia frontal (uma por estudo) para predição e erro; o paciente para divisão e incerteza; o par modelo × base para a generalização |
| Hipótese principal | Padronizar as características de qualidade reduz a queda de AUROC entre fontes além do que a padronização clínica já reduz |

---

## 1. O problema

**Classificadores de pneumonia costumam funcionar pior fora da instituição onde foram treinados, e ainda não se sabe quanto dessa queda vem da imagem em si.**

Evidência que motiva o estudo:

- **A queda externa é comum.** Em 3 de 5 comparações entre hospitais, o desempenho externo foi significativamente menor que o interno (Zech et al., 2018).
- **Há explicações concorrentes.** O significado do rótulo muda entre bases (Cohen et al., 2020), e a intensidade das imagens também (Pooch et al.).
- **Parâmetros de aquisição podem virar atalho.** A AUC para pneumotórax caiu de 0,94 para 0,56 quando as distribuições de exposição foram invertidas entre treino e teste (Shu et al., 2026).
- **O efeito também pode ser pequeno.** Sob redução de dose simulada, a AUROC mudou menos de 0,005 em média, e a posição do paciente pesou mais que a dose (Erenstein et al., 2025).

**A lacuna candidata.** Numa busca estruturada de escopo (47 consultas, sem revisão sistemática formal), não apareceu estudo que junte ao mesmo tempo:

1. qualidade natural medida na imagem;
2. parâmetros físicos de aquisição;
3. erros **e** calibração para pneumonia;
4. teste em instituição independente;
5. a parcela da queda externa associada à qualidade.

Isso é uma lacuna candidata, não prova de ineditismo. A revisão sistemática formal segue como pendência.

**Por que importa:** se a queda vem da qualidade da imagem, ela pode ser monitorada e padronizada. Se vem do rótulo, da população ou da prevalência, a resposta é outra (recalibrar, retreinar, rever a referência). Confundir as duas leva a correções erradas.

---

## 2. A pergunta definida

### 2.1 Pergunta principal

> Em fontes clínicas independentes, em que medida diferenças nas distribuições de características naturais de qualidade e de parâmetros de aquisição estão associadas à diferença de erro e de calibração de um classificador de pneumonia, após padronização de fatores clínicos, projeção e referência?

### 2.2 Como cada parte da pergunta é respondida

| Parte da pergunta | Como vira análise |
|---|---|
| "fontes clínicas independentes" | Três origens distintas: Stanford (CheXpert Plus), NIH (RSNA) e Hospital de San Juan, Alicante (PadChest), sem pacientes em comum |
| "características naturais de qualidade" | Oito medidas na região do pulmão: brilho, contraste, saturação e enquadramento (primárias); amplitude, entropia, nitidez e ruído (secundárias) |
| "parâmetros de aquisição" | kVp, corrente, tempo, produto corrente-tempo e índices de exposição, só dentro do mesmo fabricante |
| "erro" | Falsos positivos e falsos negativos contra a referência de cada base, com limiar fixado na validação da fonte |
| "calibração" | ECE com 15 faixas de igual tamanho, intercepto e inclinação de calibração |
| "após padronização de fatores clínicos, projeção" | Pesos de importância por posição, idade e sexo (padronização S1) |
| "e referência" | Controles de rótulo: alvo alternativo, subconjunto com baixa discordância entre rotuladores, subconjunto rotulado por médicos |

### 2.3 O que a pergunta **não** pergunta

- **Causa.** O estudo mede associação e padronização. A "parcela da queda" é a redução do gap sob uma padronização explícita, não uma fração causal.
- **Uso clínico.** Nenhum resultado serve para diagnóstico ou cuidado de pacientes. A licença do PadChest proíbe isso expressamente.
- **Qual arquitetura é melhor.** O modelo é fixo (DenseNet121 com pesos ImageNet), para que as diferenças venham dos dados.

---

## 3. A decisão que o estudo informa

**O estudo informa uma decisão de método: na validação externa de um classificador de radiografia, vale medir e padronizar a qualidade da imagem, ou a queda se explica melhor por população, rótulo e prevalência?**

| Resultado | Decisão que ele apoia |
|---|---|
| A qualidade explica parte relevante da queda (cenário S1) | Incluir a auditoria de qualidade e de apresentação na validação externa; considerar padronização de aquisição antes de levar o modelo a outra instituição |
| Efeito pequeno ou heterogêneo (S2) | A qualidade medida não é o principal fator; o esforço de validação vai para população e rótulo |
| Atalho dominante (S3) | O classificador explora origem ou exposição; os dados precisam ser preparados antes do treino |
| Queda não explicada pela qualidade (S4) | Rótulo, população ou prevalência são hipóteses mais fortes; investigar a referência |
| Inconclusivo (S5) | Reportar a falta de poder e as estimativas com intervalo; o desenho serve de base para um estudo maior |

**Por que esta decisão:** é a que muda o que se faz na prática de validação. A literatura mostra a queda; falta saber onde investir para entendê-la.

---

## 4. Público

| Público | O que tira do estudo | Como o estudo fala com ele |
|---|---|---|
| Pesquisadores e desenvolvedores de IA em radiologia (principal) | Um protocolo reprodutível de auditoria de dados e de decomposição da queda externa | Código, listas de IDs com hash e tabelas de resultados agregados |
| Física médica e qualidade de imagem | Quais características (brilho, saturação, enquadramento) acompanham os erros, por fabricante | Razões de chances por 1 desvio-padrão; curvas dose-resposta das degradações |
| Quem avalia levar um modelo para outra instituição | Que sinais de qualidade e de apresentação monitorar | Tabelas de desempenho interno e externo com intervalo de confiança; limitações declaradas |
| Banca e orientação da iniciação científica | Rigor, transparência e limites das conclusões | Protocolo congelado antes de abrir o teste; desvios registrados |

**Quem não é o público:** pacientes e decisões clínicas individuais. Nenhuma base foi aprovada para uso clínico.

---

## 5. Unidade de análise

**A unidade muda com a pergunta de cada etapa, e misturá-las geraria conclusões erradas.**

| Nível | Unidade | Usada para | Por quê |
|---|---|---|---|
| Predição e erro | **Radiografia frontal, uma por estudo** | Probabilidade, falso positivo, falso negativo, medidas de qualidade | O modelo decide por imagem, e o rótulo é por imagem (RSNA) ou por estudo (Plus, PadChest). Uma frontal por estudo evita contar o mesmo rótulo duas vezes |
| Divisão e incerteza | **Paciente** | Divisão treino/validação/teste 70/10/20; bootstrap de 2.000 reamostragens; erro-padrão agrupado | Imagens do mesmo paciente são correlacionadas: até 91 imagens por paciente no Plus e 75 no RSNA. Dividir por imagem vazaria o paciente entre treino e teste |
| Generalização | **Par modelo × base** | Desempenho interno e externo, gap bruto, gap padronizado | A pergunta é sobre a diferença entre fontes |
| Variação do treino | **Semente** (5 por modelo) | Dispersão entre sementes, reportada separada do bootstrap | Separa a incerteza dos dados da incerteza do treino |
| Efeito da qualidade | **1 desvio-padrão da medida, padronizada dentro de cada base** | Razões de chances do eixo A | Torna as oito medidas comparáveis entre si sem misturar escalas de bases diferentes |

**Contagem de positivos:** por imagem, com intervalo de confiança por bootstrap de paciente e uma análise de sensibilidade em que cada paciente pesa o mesmo. Justificativa: o mesmo paciente pode ter a doença em fases diferentes, e o PadChest já é contado por estudo.

---

## 6. Hipóteses

### 6.1 Hipótese principal (testável)

**Escolhida porque responde diretamente à pergunta principal:** a pergunta é sobre a diferença entre fontes, e esta hipótese mede quanto dessa diferença acompanha a qualidade.

> **H2 — Padronizar as características de qualidade comparáveis entre bases reduz a queda de AUROC entre a fonte e a base externa além do que a padronização clínica (posição, idade e sexo) já reduz.**

Formalmente, para cada par fonte → externa:

```latex
D_{\text{bruto}} = \text{AUROC}_{\text{interno}} - \text{AUROC}_{\text{externo}}
```

```latex
\text{Parcela de qualidade} = \frac{D_{S1} - D_{S2}}{D_{\text{bruto}}}
```

- D_S1 = gap depois de padronizar posição, idade e sexo.
- D_S2 = gap depois de padronizar também as medidas de qualidade comparáveis.
- **H0:** D_S1 − D_S2 = 0 (a qualidade não acrescenta nada à padronização clínica).
- **H2:** D_S1 − D_S2 > 0.

**Regras de leitura, fixadas antes de ver o teste:**

1. Reportar as diferenças absolutas antes da razão.
2. Se D_bruto estiver perto de zero, não calcular a parcela.
3. Não truncar valores negativos nem acima de 100%.
4. Reestimar os pesos em cada reamostragem do bootstrap.

### 6.2 Hipóteses complementares

| Hipótese | Enunciado testável | Teste |
|---|---|---|
| **H1** — qualidade e erro | Dentro de cada base, pelo menos uma das 4 medidas primárias de qualidade está associada a falsos negativos (entre positivos) ou a falsos positivos (entre negativos), depois de ajustar por posição, idade, sexo e fabricante | Regressão logística, razão de chances por 1 desvio-padrão, erro-padrão agrupado por paciente; correção de Holm nas 4 primárias |
| **H3** — calibração ≠ discriminação | O efeito da qualidade sobre a calibração (ECE) difere do efeito sobre a discriminação (AUROC) | Comparar, no mesmo nível de degradação ou na mesma padronização, ΔECE e ΔAUROC frente às respectivas margens, com bootstrap pareado |
| **H4** — regra de leitura | O resultado pode ser nulo ou heterogêneo; o protocolo não pressupõe que a qualidade seja a principal explicação | Não é teste: um resultado nulo dentro das margens é reportado como achado |

**Teste de coerência (T3):** uma associação natural (eixo A) só é lida como robusta se as degradações controladas (eixo B) mostrarem tendência monotônica na mesma direção, em pelo menos 2 dos 3 desfechos, com o controle negativo (percentis 25/75) sem efeito.

---

## 7. Critério de sucesso

### 7.1 Critério de apoio à hipótese principal

A hipótese principal (H2) é **apoiada** num par fonte → externa quando as três condições valem juntas:

| Condição | Valor |
|---|---|
| Significância | Intervalo de confiança de 95% de D_S1 − D_S2, por bootstrap de paciente, exclui zero, depois da correção de Holm na família de testes primários |
| Relevância | D_S1 − D_S2 ≥ **0,02 de AUROC** em valor absoluto (margem proposta) |
| Coerência | Mesma direção no eixo B (T3) |

Efeitos significativos abaixo da margem são reportados como **nulos na prática**. As margens propostas também valem para ΔECE ≥ 0,02 e ΔFNR ou ΔFPR ≥ 0,03, e serão confirmadas no congelamento.

**Por que estas margens:** separam significância estatística de relevância prática. Com dezenas de milhares de imagens, diferenças minúsculas ficam significativas sem mudar nada na prática.

### 7.2 Critério de sucesso do projeto

**Independente do cenário que aparecer:** todas as análises pré-especificadas executadas, reprodutíveis, reportadas com incerteza e com as limitações declaradas.

Na prática, isso exige:

1. Protocolo congelado com hash antes de qualquer leitura do teste.
2. Teste destrancado só depois do treino final, com data, hora e hashes registrados.
3. Cada par modelo × base classificado num cenário (S1 a S5) antes de escrever a discussão.
4. Retreino de um modelo (M1, semente 0) num ambiente limpo, dentro da tolerância registrada.

**Por que o sucesso não é "a hipótese se confirmar":** um resultado nulo bem medido também responde à pergunta e também orienta a decisão (seção 3).

**O desempenho do modelo não é critério de sucesso.** Ele precisa apenas passar nas verificações de sanidade: rótulos embaralhados levam a AUROC perto de 0,5 na validação, e execuções idênticas dão o mesmo resultado dentro da tolerância.

---

## 8. Como a auditoria dos dados refinou a pergunta

A auditoria (etapa 02) não mudou a pergunta, mas mudou o peso de algumas partes. Cada ajuste abaixo é proposta para o congelamento.

| Achado da auditoria | Ajuste |
|---|---|
| Parâmetros físicos existem em só cerca de 20% do Plus (quase todos em PA da GE); nenhum no RSNA; PadChest sem kVp | Os parâmetros físicos viram análise **exploratória**, dentro do fabricante. A análise principal usa as medidas de qualidade da imagem |
| As imagens das três bases já chegam processadas (RSNA: PNG de 8 bits e depois JPEG com perda; PadChest: reescalado para 16 bits; Plus: proveniência não comprovável) | A redação passa a ser "qualidade e apresentação **nas imagens distribuídas**", não "na aquisição" |
| A projeção se associa ao rótulo em direções opostas: RSNA 37,9% de pneumonia no AP contra 9,2% no PA; Plus 1,72% contra 3,19% | Reforça a padronização por posição (S1) e o teste de atalhos (T1). É uma explicação concreta e testável para a queda entre bases |
| Fabricante, saturação e prevalência andam juntos no PadChest (Imaging Dynamics: 59% das imagens com saturação e 1,24% de pneumonia; Philips: 0% e 3,90%) e fabricante e projeção andam juntos no Plus | Análises de qualidade sempre dentro do fabricante; declarar que, no Plus, posição e fabricante não se separam |
| Os rótulos medem coisas diferentes: opacidade vista pelo radiologista (RSNA), menção no laudo extraída por algoritmo (Plus), rótulo de médico a partir do laudo (PadChest) | Os controles de rótulo deixam de ser complemento e passam a ser parte central da leitura da queda |
| O Plus rende cerca de 689 positivos no teste, abaixo de 750 | O Plus usa 2 faixas de qualidade (mínimo de 500 positivos), não 3 |
| A máscara do pulmão tende a deixar de fora o pulmão opacificado | Testar se a máscara encolhe nos positivos antes do congelamento; máscara central fixa como sensibilidade |

---

## 9. Escopo

| Dentro | Fora nesta versão | Fora em definitivo |
|---|---|---|
| Pneumonia em adultos (18 a 100 anos) | Laudo e imagem juntos (multimodal) — v2 | Segunda arquitetura |
| Radiografias frontais AP e PA, uma por estudo | Pediatria (PediCXR, Kermany) — v3 | Afirmações causais |
| Três bases de origens distintas | Outros achados além de pneumonia | Conclusões sobre o sinal cru sem prova de processamento mínimo |
| Um modelo fixo (DenseNet121, pesos ImageNet) | — | Uso clínico |
| Associação e padronização, com incerteza | — | — |

---

## 10. Definições operacionais

| Termo | Definição |
|---|---|
| Qualidade natural | Propriedades observadas nas imagens como distribuídas, antes de qualquer perturbação, medidas na região do pulmão. "Natural" não significa sem processamento do fabricante |
| Apresentação | Posição (AP/PA), conversão para 8 bits e janelamento aplicados à imagem distribuída |
| Parâmetros físicos | Valores do cabeçalho (kVp, corrente, tempo, produto corrente-tempo, índices de exposição), com unidades e validade verificadas |
| Erro | Falso positivo ou falso negativo contra a referência de cada base, com o limiar fixado na validação da fonte. Não é necessariamente erro clínico |
| Calibração | Correspondência entre probabilidade prevista e frequência observada (ECE, curva de confiabilidade, intercepto, inclinação) |
| Fonte externa independente | Outra origem clínica, sem pacientes ou imagens de teste no desenvolvimento. Outro nome de dataset não basta |
| Parcela da queda associada à qualidade | Redução padronizada do gap externo sob uma padronização explícita. Não é fração causal |

---

## Referências

1. Zech JR et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. *PLOS Medicine*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/
2. Pooch EHP, Ballester PL, Barros RC. Can we trust deep learning models diagnosis? The impact of domain shift in chest radiograph classification. https://arxiv.org/abs/1909.01940
3. Cohen JP et al. (2020). On the limits of cross-domain generalization in automated X-ray prediction. *MIDL/PMLR* 121. https://proceedings.mlr.press/v121/cohen20a.html
4. Shu HJ et al. (2026). Impact of exposure parameters on deep learning models in chest radiography and implications for deployment. *Radiology: Artificial Intelligence* 8(4). https://doi.org/10.1148/ryai.250731
5. Erenstein H et al. (2025). The effect of simulated dose reduction on the performance of artificial intelligence in chest radiography. *Journal of Imaging* 11(3), 90. https://doi.org/10.3390/jimaging11030090
6. Protocolo científico v1.0 e Plano de execução do projeto (17/10/2026).
