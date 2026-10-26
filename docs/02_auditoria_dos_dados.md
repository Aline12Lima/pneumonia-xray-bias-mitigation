# 02 — Auditoria dos dados

**Projeto:** Qualidade e apresentação de radiografias de tórax e erros, calibração e generalização de classificadores de pneumonia
**Etapa do ciclo:** 2 de N (auditoria dos dados; fase F3 do plano de execução)
**Período de execução:** 17/10 a 22/10/2026
**Versão:** 23/10/2026 · Aline Lima
**Documentos de detalhe:** `auditorias/chexpert_plus_final.md`, `auditorias/rsna_final.md`, `auditorias/padchest_final.md`

---

## Resumo

**As três bases foram aprovadas para entrar no estudo.** Todas têm positivos suficientes no teste, ID de paciente utilizável, máscara do pulmão aceitável em pelo menos 95 de 100 imagens e termos de uso compatíveis. O Plus entra com ressalvas.

| | CheXpert Plus | RSNA Pneumonia | PadChest |
|---|---|---|---|
| Origem | Stanford (EUA) | NIH Clinical Center (EUA) | Hospital de San Juan, Alicante (Espanha) |
| Papel no estudo | Treino (M1, M3); teste interno; externa do M2 | Treino (M2, M3); teste interno; externa do M1 | Só teste externo (M1, M2, M3) |
| Imagens na base final | 176.660 | 25.403 | 24.770 |
| Pacientes | 62.832 | cerca de 10.926 | cerca de 22.987 |
| Positivos | 3.444 (1,95%) | 5.627 (22,1%) | 796 (3,21%) |
| Positivos esperados no teste | cerca de 689 | cerca de 1.125 | 796 (toda a base é teste) |
| Máscara do pulmão aceitável | 95/100 | 100/100 (97/100 na leitura estrita) | 97/100 |
| **Situação** | **Aprovada com ressalvas** (2 faixas de qualidade) | **Aprovada** | **Aprovada** |

Os três achados que mais pesam no estudo:

1. **A projeção se associa à pneumonia em direções opostas.** RSNA: 37,9% no AP contra 9,2% no PA. Plus: 1,72% contra 3,19%. PadChest: 6,1% contra 2,95%.
2. **Fabricante, qualidade e prevalência andam juntos.** No PadChest, o Imaging Dynamics satura o branco em 59% das imagens e tem 1,24% de pneumonia; o Philips, 0% e 3,90%. No Plus, o AP é 99,3% Agfa e o PA, 87,2% GE.
3. **Nenhuma base traz a imagem "crua".** Todas chegam processadas, de formas diferentes. As medidas de qualidade descrevem as imagens como foram distribuídas.

---

## 1. Origem dos dados

**Cada base vem de uma instituição diferente, o que é a condição para chamar uma delas de externa em relação às outras.**

| Base | Instituição | Período das imagens | Quem rotulou | Referência |
|---|---|---|---|---|
| CheXpert Plus | Stanford Hospital, pacientes internados e ambulatoriais | Outubro de 2002 a julho de 2017 | Algoritmo (CheXbert) aplicado à seção de impressão do laudo | Chambon et al., 2024 |
| RSNA Pneumonia (Stage 2) | NIH Clinical Center, via NIH ChestX-ray8 | 1992 a 2015 (base NIH) | Radiologistas da Society of Thoracic Radiology, com adjudicação | Shih et al., 2019; Wang et al., 2017 |
| PadChest | Hospital Universitario de San Juan, Alicante | 2009 a 2017; no recorte, 76% de 2015 a 2017 | Médicos, a partir do laudo (subconjunto rotulado manualmente) | Bustos et al., 2020 |

**Independência:** as três são origens clínicas distintas. O RSNA é um subconjunto do NIH, por isso o NIH completo foi excluído do estudo. O CheXpert JPG também ficou de fora: são as mesmas imagens do Plus, só que com equalização de histograma, que apagaria justamente o brilho e o contraste que o estudo mede.

---

## 2. Acesso

**Os três acessos estão resolvidos, e todos os dados ficam em espaços privados no Redivis, acessíveis só à autora.**

| Base | Como foi obtida | Onde está | Situação |
|---|---|---|---|
| CheXpert Plus | Dataset "CheXpert Plus v1.0" já hospedado no Redivis, com exportação restrita | Workflow `auditoria_chexpert_plus` | Os notebooks não têm internet; pacotes entram só pelas dependências |
| RSNA | Distribuição oficial da RSNA (S3 público), baixada direto num notebook do Redivis. Substituiu a cópia do Kaggle | Dataset próprio `rsna_pneumonia_2018` v1.0 (30.000 DICOM e tabela de rótulos) e workflow `auditoria_rsna` | Acesso restrito à autora |
| PadChest | Pedido de acesso aprovado pelo BIMCV. Pacotes baixados um por vez dentro do notebook, processados e apagados | Dataset privado `padchest` (só o CSV de rótulos); cópias de 320 px em `/out` do workflow `auditoria_padchest` | Nenhuma imagem original fica guardada; o link de download não é compartilhado |

**Por que no Redivis:** o Plus já está lá, os dados ficam num ambiente controlado, e o processamento em fluxo (baixar, conferir, guardar o derivado, apagar) evita guardar terabytes.

**Espaço ocupado:** cerca de 8,5 a 12,5 GB dos 30 GB da conta (4 GB do RSNA, 4,4 GB do PadChest, 0,1 GB de tabelas; mais 4 GB se o zip do RSNA ainda não tiver sido apagado).

---

## 3. Formato

**Nenhuma das três bases entrega a imagem como saiu do aparelho, e cada uma foi processada de um jeito diferente.**

| | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| Formato usado | DICOM | DICOM (*secondary capture*) | PNG |
| Profundidade | 16 bits alocados; 12, 14 ou 15 armazenados | 8 bits | 16 bits |
| Resolução | Variável (por exemplo 2021 × 2021 a 2800 × 3408) | 1024 × 1024 em todas | Variável (mediana 2000 × 2000 no Imaging Dynamics; 2545 × 2706 no Philips) |
| Compressão | Sem perda | **JPEG com perda** em 100% | Sem perda (PNG) |
| Escala de cinza | MONOCHROME1 na Agfa (invertida); MONOCHROME2 na GE | MONOCHROME2 | MONOCHROME2 |
| Janela de visualização | Só na GE | Ausente | No CSV, mas em outra escala (ver abaixo) |
| Parâmetros físicos | Em cerca de 20% da base, quase só GE | Ausentes | Corrente e exposição no Imaging Dynamics; sem kVp |
| Fabricante | Agfa (cerca de 86%) e GE (cerca de 14%) | Ausente | Philips (74%) e Imaging Dynamics (26%) |

**Achados de formato que mudam a análise:**

- **RSNA processado duas vezes.** O NIH reduziu as radiografias para PNG de 8 bits, e a RSNA regravou essas imagens como DICOM com JPEG com perda. Por isso o RSNA fica fora da análise com parâmetros físicos, e nitidez e ruído só são comparados dentro da base.
- **PadChest reescalado.** O CSV diz 10 ou 12 bits, mas os PNG usam toda a escala de 16 bits: os valores foram esticados linearmente (12 bits multiplicados por 16,004; 10 bits por 64,06). A profundidade original se mantém (no máximo cerca de 3.700 tons diferentes por imagem). A janela do CSV só funciona depois dessa conversão: passa a cobrir 100% da imagem no Philips e 69% no Imaging Dynamics, contra 0% a 8% na escala original.
- **Plus sem prova de proveniência.** Os elementos do cabeçalho que provariam imagem não processada estão ausentes, e `RescaleType = P-VALUES` sugere valores de apresentação. O PNG público foi descartado porque teve equalização de histograma.

**Conversão para 8 bits (proposta, igual nas três bases):** mínimo-máximo por percentis 0,5–99,5, com inversão das imagens MONOCHROME1. Ela foi escolhida porque o mínimo-máximo puro "lava" as imagens com pixels extremos, e porque uma regra única evita que a conversão acompanhe o fabricante. A janela da GE (Plus) e a janela reescalada (PadChest) entram como análises de sensibilidade.

---

## 4. Licença e termos de uso

**Os três termos permitem o estudo, mas o PadChest impõe as condições mais estritas.**

| | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| Uso permitido | Só pesquisa, sem uso comercial | Pesquisa, ensino e outros fins, comerciais ou não | Só pesquisa acadêmica; outros usos precisam de aprovação |
| Redistribuição | Proibida | Não proibida; exige atribuição | **Proibida**, inclusive de qualquer parte, sem permissão por escrito |
| Dados derivados | Guardar só derivados | Não tratado | Proibidos, exceto modelos e algoritmos |
| Link de download | — | Público | **Não pode ser compartilhado** fora da organização |
| Reidentificação | Proibida | Proibida | Proibida |
| Atribuição | Chambon et al., 2024 | Wang et al., 2017 (com link e agradecimento ao NIH Clinical Center); Shih et al., 2019 | Bustos et al., 2020 |

**Consequências práticas:**

1. Todos os datasets e workflows ficam **restritos à autora**.
2. **Não se publica nenhum dado por imagem do PadChest.** Só resultados agregados (tabelas, métricas, figuras) e o código. Isso altera o passo 12.4 do plano, que previa arquivar features derivadas com DOI.
3. As cópias de 320 px e as medidas de qualidade servem só para avaliar o modelo e ficam internas.
4. Para o RSNA, foi adotada a fonte oficial: os termos da RSNA são compatíveis, enquanto as regras da competição do Kaggle não chegaram a ser conferidas.

---

## 5. Volumes

**O volume bruto passa de 4 TB; o que entra no estudo são cerca de 227 mil radiografias.**

### 5.1 Volume bruto

| | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| Imagens | 223.462 | 30.000 | 160.861 |
| Estudos | 187.711 | — | 109.931 |
| Pacientes | 64.725 | 12.274 | 67.625 |
| Tamanho | 2,58 TB (DICOM soltos) | 3,98 GB | cerca de 1 TB (52 pacotes); 282 GB nos 13 do recorte |

### 5.2 Do bruto à base final

**CheXpert Plus**

| Etapa | Imagens | Pacientes | Positivos | Saíram |
|---|---|---|---|---|
| Total | 223.462 | 64.725 | 4.847 | — |
| Frontais | 191.071 | 64.710 | 3.731 | 32.391 laterais |
| AP ou PA | 191.054 | 64.710 | 3.730 | 17 (LL ou RL) |
| Com DICOM | 180.073 | 62.933 | 3.524 | 10.981 sem DICOM |
| Idade de 18 anos ou mais, registrada | 179.853 | 62.833 | 3.522 | 220 |
| Uma frontal por estudo | 176.670 | 62.833 | 3.444 | 3.183 |
| Sem duplicata exata | 176.661 | 62.833 | 3.444 | 9 |
| Sexo registrado | **176.660** | **62.832** | **3.444** | 1 |

**RSNA**

| Etapa | Imagens | Pacientes | Positivos | Saíram |
|---|---|---|---|---|
| Pacote oficial | 30.000 | 12.274 | — | — |
| Com rótulo final | 29.684 | 12.233 | 7.106 | 316 (106 "Exclude", 210 sem rótulo) |
| Só o treino do Kaggle | 26.684 | 11.452 | 6.012 | 3.000 do teste do Kaggle (guardadas à parte) |
| Idade de 18 a 100 anos | **25.403** | **cerca de 10.926** | **5.627** | 1.276 menores e 5 acima de 100 |

**PadChest**

| Etapa | Imagens | Estudos | Pacientes | Positivos (por estudo) |
|---|---|---|---|---|
| Todas | 160.861 | 109.931 | 67.625 | 5.037 |
| Pacotes 0–11 e 54 (rotulados por médicos) | 38.606 | 26.112 | 24.217 | 1.054 |
| Frontais (PA, AP, AP deitado) | 26.126 | 25.786 | 23.965 | 1.026 |
| Idade de 18 a 100 anos | 25.129 | 24.830 | 23.037 | 796 |
| Sem rótulo "exclude" | 25.072 | 24.773 | 22.987 | 796 |
| Uma frontal por estudo e sexo registrado | **24.770** | **24.770** | **cerca de 22.987** | **796** |

---

## 6. Rótulos

**Os três rótulos medem coisas diferentes, e isso é uma explicação concorrente para a queda entre bases.**

| | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| O que o rótulo mede | Menção a pneumonia na **impressão** do laudo | **Opacidade pulmonar sugestiva de pneumonia** vista na imagem | Pneumonia registrada no laudo, rotulada por médico |
| Quem rotulou | Algoritmo CheXbert | Radiologistas, com adjudicação | Médicos (100% do recorte) |
| Categorias | Positivo, negativo, incerto, não mencionado | Lung Opacity; No Lung Opacity / Not Normal; Normal | Lista de achados por estudo |
| Positivo | "Pneumonia" = 1 | Lung Opacity | Rótulo "pneumonia" |
| Negativo | Negativo explícito e não mencionado | Normal e Not Normal (classe guardada para sensibilidade) | Ausência de "pneumonia" |
| Política de incerto | **Excluído** na análise principal; = positivo na sensibilidade | Não há incerto | Não há incerto |
| Sensibilidade | Rótulo da seção de achados | As 3.000 imagens do teste do Kaggle (rótulo confirmado por 3 radiologistas) | Incluir "atypical pneumonia" (812 positivos) |

**Achados de rótulo:**

- **O arquivo de rótulos do laudo inteiro do Plus está desalinhado.** Seus rótulos não correspondem ao texto da imagem da mesma linha: a fração de rótulos marcados é igual quando o laudo menciona ou não o achado (por exemplo, pneumonia: 15,3% contra 15,0%). Os arquivos de impressão e de achados estão alinhados (99,9% contra 7,4%). Por isso o Plus usa a impressão.
- **As contagens de rótulo citadas no protocolo** (6.042 positivos no Plus) vêm de outro rotulador, o CheXpert, que não está disponível no Redivis. Na base final, o Plus tem 3.444 positivos pelo CheXbert da impressão.
- **O rótulo do Plus é o mais fraco contra a imagem.** Segundo o artigo, o CheXbert da impressão tem F1 de 0,34 contra radiologistas olhando a imagem e 0,93 contra o texto (média dos 14 achados).
- **O teste do Kaggle do RSNA é enriquecido:** 36,5% de positivos contra 22,5% no treino, com outro processo de rotulagem. Por isso fica fora da análise principal.

---

## 7. Ausências e duplicatas

### 7.1 Ausências

| Base | O que falta | Quanto | Tratamento | Enviesa? |
|---|---|---|---|---|
| Plus | DICOM da imagem | 12.493 imagens (5,6%) | Excluídas | Não: falta para 5,52% dos positivos e 5,75% dos não mencionados |
| Plus | Idade | 217 imagens | Excluídas (idade é covariável) | Pouco provável (0,1%) |
| Plus | Sexo | 1 imagem | Excluída | Não |
| Plus | Parâmetros físicos | Cerca de 80% da base (quase toda a Agfa) | Análise física só exploratória, dentro da GE | Sim, se generalizada: restrita ao subgrupo |
| Plus | Janela de visualização | Toda a Agfa | Conversão por percentis para todas | Não, com regra única |
| Plus | Segundo rotulador (CheXpert) | Toda a base | **Pendente:** baixar o CSV do CheXpert 1.0 | Limita o controle de discordância |
| RSNA | Rótulo final | 316 imagens | Excluídas | Não: ficaram fora do desafio |
| RSNA | Fabricante, janela, parâmetros físicos | Toda a base | Fora da análise física | — |
| RSNA | ID de paciente no DICOM | Toda a base (o campo repete o ID da imagem) | Paciente pelo nome original no NIH | Não |
| PadChest | kVp | Toda a base | Análise física restrita ao que existe, dentro do fabricante | — |
| PadChest | Sexo registrado | 3 imagens (sexo "O") | Excluídas | Não (nenhum positivo) |
| PadChest | Imagens que não abrem | 0 de 24.773 | — | — |

### 7.2 Duplicatas

| Base | Duplicatas exatas (MD5) | Tratamento |
|---|---|---|
| Plus | 6 grupos, 12 arquivos: 3 grupos entre **pacientes diferentes** e 3 no mesmo paciente em estudos consecutivos | Excluídos todos os membros dos grupos entre pacientes (vínculo imagem–laudo duvidoso); no mesmo paciente, mantido o estudo mais antigo. Saíram 9 imagens |
| RSNA | 0 | — |
| PadChest | 0 em 24.773 | — |

**Pendente nas três bases:** duplicatas por pixel (hash perceptual), que pegam a mesma imagem regravada com outro arquivo. Ficam para a passada completa das medidas de qualidade.

---

## 8. Grupos presentes

**Prevalência de pneumonia por grupo, na base final de cada base.**

### 8.1 Projeção

| Projeção | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| AP | 149.341 imagens · **1,72%** | 11.459 · **37,9%** | 2.060 · **6,1%** (AP 8,5%; AP deitado 5,6%) |
| PA | 27.319 · **3,19%** | 13.949 · **9,2%** | 22.713 · **2,95%** |

### 8.2 Sexo

| Sexo | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| Feminino | 73.794 · 1,90% | 11.021 · 21,3% | 12.956 · 2,86% |
| Masculino | 102.866 · 1,98% | 14.387 · 22,8% | 11.814 · 3,60% |

### 8.3 Idade

| Faixa | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| 18–39 | 24.525 · 2,12% | 7.248 · 24,2% | 2.024 · 5,39% |
| 40–59 | 53.878 · 1,67% | 11.612 · 21,7% | 8.487 · 2,45% |
| 60–74 | 53.942 · 1,76% | 5.720 · 20,2% | 8.371 · 3,07% |
| 75 ou mais | 44.315 · 2,43% | 828 · 22,9% | 5.891 · 3,77% |

### 8.4 Fabricante

| Fabricante | CheXpert Plus (amostra de 2.000 cabeçalhos) | PadChest |
|---|---|---|
| Agfa | Cerca de 86% da base; positivos na amostra 1,6% | — |
| GE | Cerca de 14% da base; positivos na amostra 2,4% a 2,7% | — |
| Philips | — | 18.340 · 3,90% |
| Imaging Dynamics | — | 6.433 · 1,24% |

O RSNA não registra fabricante.

### 8.5 Imagens por paciente

| | CheXpert Plus | RSNA | PadChest |
|---|---|---|---|
| Mediana | 1 | 1 | 1 |
| Média | 2,8 | cerca de 2,3 | cerca de 1,1 |
| Máximo | 91 | 75 | — |

---

## 9. Representatividade

**Cada base representa a população de um hospital, num período, com o seu próprio processo de rotulagem. Nenhuma representa "a radiografia de tórax" em geral.**

| Dimensão | O que as bases cobrem | O que fica de fora ou sub-representado |
|---|---|---|
| Geografia | Dois hospitais dos EUA e um da Espanha | América Latina, Ásia, África; hospitais de pequeno porte |
| Idade | Adultos de 18 a 100 anos | Crianças (excluídas: 230 estudos positivos saíram só no PadChest). No RSNA, só 828 imagens a partir de 75 anos (3%), contra 25% no Plus |
| Prevalência | De 1,95% (Plus) a 22,1% (RSNA) | O RSNA foi montado com enriquecimento de casos, então sua prevalência não é a de um pronto-socorro |
| Gravidade e local | Plus e RSNA com muito exame de leito (AP); PadChest quase todo PA (92%), de pé | Contexto clínico do exame não é registrado |
| Equipamento | Agfa, GE, Philips, Imaging Dynamics | No RSNA, o equipamento é desconhecido; outros fabricantes ausentes |
| Período | 1992 a 2017 | Equipamentos e protocolos atuais |
| Raça e etnia | Registradas no Plus | **Não auditadas** nesta etapa; ausentes no RSNA e no PadChest |
| Rótulo | Três definições diferentes de "pneumonia" | Diagnóstico clínico confirmado (exame laboratorial, evolução) |

**Consequência para a leitura dos resultados:** as conclusões valem para estas três instituições e estas imagens distribuídas. A generalização para outros hospitais é exatamente o que o estudo mede, não o que ele pressupõe.

---

## 10. Riscos de vazamento

**Vazamento é qualquer caminho pelo qual informação do teste chega ao treino, inflando o desempenho. A tabela separa os riscos já controlados dos que ainda dependem de passos futuros.**

| Risco | Onde aparece | Proteção | Situação |
|---|---|---|---|
| Mesmo paciente no treino e no teste | Plus (até 91 imagens por paciente), RSNA (até 75) | Divisão **por paciente** (70/10/20, semente 2026); verificação de que nenhum paciente aparece em duas partições | Planejada (geração das listas na F4) |
| Mesma imagem em pacientes diferentes | Plus: 3 grupos | Exclusão de todos os membros | **Controlado** |
| Mesma imagem regravada | Todas | Hash perceptual | **Pendente** |
| Pacientes em comum entre bases | Três origens distintas | Exclusão do NIH completo e do CheXpert JPG | **Controlado** |
| Conjunto de sensibilidade do RSNA com pacientes do treino | As 3.000 do teste do Kaggle: 1.575 pacientes também estão no treino do Kaggle | Ao usar essas 3.000, excluir os pacientes que caírem no treino da nossa divisão | **A aplicar** quando o conjunto for usado |
| Pesos pré-treinados que viram as bases | DenseNet121 | Pesos ImageNet, sem radiografias; pesos do TorchXRayVision descartados | **Controlado** |
| Modelo de máscara que viu as bases | Segmentação do pulmão | `lungs-segmentation`, treinado só em Montgomery e Shenzhen. O CheXmask foi descartado porque viu imagens do PadChest | **Controlado** |
| Estatísticas calculadas com o teste | Conversão para 8 bits, normalização | Conversão por imagem (percentis da própria imagem); normalização fixa do ImageNet | **Controlado** |
| Limiar e calibração ajustados no teste | Limiar e temperatura | Ajustados só na validação da fonte | Planejado |
| Decisões tomadas olhando o teste | Todo o protocolo | Listas com hash, teste trancado, código de análise congelado antes de abrir o teste | Planejado (F4 e F6) |
| Texto do laudo no rótulo | Plus | O laudo não é entrada do modelo na v1 | **Controlado** |

**Atalhos que não são vazamento, mas podem inflar o desempenho interno e derrubar o externo:**

- **Projeção:** a prevalência muda de direção entre bases (seção 8.1). Medido pelo teste T1 contra uma linha de base só com a projeção.
- **Texto gravado na imagem:** algumas imagens AP do RSNA têm "PORTABLE" escrito no pixel.
- **Fabricante e saturação:** no PadChest, o Imaging Dynamics satura o branco e tem um terço da prevalência do Philips.
- **Máscara que encolhe na doença:** o modelo de máscara tende a deixar de fora o pulmão opacificado. Se isso acontecer mais nos positivos, as medidas de qualidade "no pulmão" passam a refletir a doença. **Testado em 26/10** no RSNA, com 1.000 imagens e regra fixada antes: a máscara não depende da doença (detalhes em `auditorias/rsna_final.md` §8.2). Fica a observação de que o brilho dentro do pulmão é maior nos positivos com qualquer máscara, porque a opacidade é clara: decisão aberta para a F4.

---

## 11. Pendências

| Pendência | Bloqueia a aprovação das bases? | Quando |
|---|---|---|
| Medidas de qualidade (as 8 features) nas três bases | Não | Código pronto (`src/qualidade/medidas.py`). Passada completa nas três bases; no PadChest, segunda passada pelos 13 pacotes (F3 › PadChest › Segunda passada) |
| Duplicatas por pixel nas três bases | Não | Na mesma passada |
| Rótulos do rotulador CheXpert no Plus | Não | Download do CSV do CheXpert 1.0 |
| Raça e etnia no Plus | Não | Descrição na composição |
| Testes de atalho exploratórios (T1, T2) em treino e validação | Não | Depois de gerar as listas de divisão |

---

## 12. Decisões desta etapa

Nomeadas por **Fase › Base › Tema**. Todas são propostas e viram definitivas no congelamento do protocolo.

| Decisão | Conteúdo |
|---|---|
| F3 › Todas › Idade | Adultos de 18 a 100 anos, com idade registrada |
| F3 › Todas › Primeira frontal | Uma frontal por estudo (Plus: menor número de vista; PadChest: PA, depois AP, depois AP deitado) |
| F3 › Todas › Duplicatas | Excluir grupos entre pacientes diferentes; no mesmo paciente, manter o mais antigo |
| F3 › Todas › Sexo | Excluir sexo não registrado |
| F3 › Todas › Contagem | Positivos por imagem; intervalos por bootstrap de paciente; sensibilidade com peso igual por paciente |
| F3 › Todas › Conversão | Percentis 0,5–99,5 nas três bases, com inversão de MONOCHROME1; janelas como sensibilidade |
| F3 › Todas › Máscara | `lungs-segmentation` resnet34, mesmo procedimento nas três bases; máscara central fixa como sensibilidade |
| F3 › Plus › Rótulo | CheXbert da impressão; incerto excluído; laudo inteiro descartado (arquivo desalinhado) |
| F3 › Plus › Exclusões | Imagens sem DICOM excluídas |
| F3 › Plus › Faixas | 2 faixas de qualidade (cerca de 689 positivos no teste) |
| F3 › Plus › Física | Parâmetros físicos só exploratórios, dentro da GE |
| F3 › Plus › Fabricante | Duas famílias (Agfa, GE); declarar que posição e fabricante não se separam |
| F3 › RSNA › Fonte | Distribuição oficial da RSNA, não o Kaggle |
| F3 › RSNA › Rótulo | Grupo final "Calculated": positivo = Lung Opacity |
| F3 › RSNA › Recorte | Treino do Kaggle na análise principal; as 3.000 do teste como sensibilidade |
| F3 › RSNA › Paciente | 8 primeiros dígitos do nome original no NIH |
| F3 › RSNA › Física | Fora da análise física; "imagem distribuída, já processada e comprimida" |
| F3 › PadChest › Termos | Dados privados; publicar só resultados agregados; link não compartilhado |
| F3 › PadChest › Recorte | Pacotes 0–11 e 54 (100% rotulados por médicos) como teste externo |
| F3 › PadChest › Escala | PNG reescalados para 16 bits; janela do CSV só depois da conversão |
| F3 › PadChest › Fabricante | Análises de qualidade sempre dentro do fabricante |

---

## Referências

1. Chambon P, Delbrouck JB et al. (2024). CheXpert Plus: augmenting a large chest X-ray dataset with text radiology reports, patient demographics and additional image formats. https://arxiv.org/abs/2405.19538
2. Shih G et al. (2019). Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. *Radiology: Artificial Intelligence*. https://doi.org/10.1148/ryai.2019180041
3. Wang X et al. (2017). ChestX-ray8: hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. CVPR.
4. Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M (2020). PadChest: a large chest x-ray image dataset with multi-label annotated reports. *Medical Image Analysis* 66, 101797. https://arxiv.org/abs/1901.07441
5. RSNA Pneumonia Detection Challenge (2018): página oficial, termos de uso e atribuição. https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
6. PadChest Dataset Research Use Agreement (BIMCV), arquivo `LICENSE.md` da distribuição.
7. Modelo de máscara: https://github.com/alimbekovKZ/lungs_segmentation e https://github.com/alimbekovKZ/lungs_segmentation_train
