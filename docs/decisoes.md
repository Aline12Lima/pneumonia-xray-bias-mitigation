# Registro de decisões

Todas as decisões do projeto num só lugar, nomeadas por **Fase › Base › Tema**.

- **Estado:** *Tomada* (vale desde já) ou *Proposta* (vira tomada no congelamento do protocolo, em 31/10/2026).
- **Antes do congelamento**, mudar uma decisão é uma nova decisão. **Depois do congelamento**, qualquer mudança é um desvio, registrado com data, motivo e impacto.
- **Número antigo:** o código usado nos chats e no plano de execução, só para rastreio. Alguns números colidiram entre chats; vale sempre o nome por tema.
- **Onde está a evidência:** a coluna "Origem" aponta o documento.

---

## Planejamento (fases 0 e 1 e plano de execução)

| Decisão | Conteúdo | Por quê | Estado | Número antigo |
|---|---|---|---|---|
| Plano › Todas › Escopo | Pneumonia em adultos, radiografias frontais, linguagem de associação (não causal) | Pergunta respondível no prazo | Tomada | D-001 |
| Plano › Todas › Modelo | DenseNet121 com pesos ImageNet, fixo nos três modelos | Padrão da literatura; pesos treinados em NIH ou CheXpert quebrariam a independência | Tomada | D-002 |
| Plano › Todas › Bases | CheXpert Plus, RSNA e PadChest. Fora: CheXpert JPG (equalizado), NIH completo (contém o RSNA), MIMIC e VinDr (credenciamento), pediatria | Três origens clínicas distintas | Tomada | D-003 |
| Plano › Todas › Divisão | Por paciente, 70/10/20, semente 2026 | Evita vazamento entre partições | Tomada | D-004 |
| Plano › Todas › Vistas | Só frontais AP e PA; uma por estudo | Não duplicar o rótulo do estudo | Tomada | D-005 |
| Plano › PadChest › Papel | Só teste externo | Uma origem nunca vista por nenhum modelo | Tomada (recorte substituído por F3 › PadChest › Recorte) | D-006 |
| Plano › Todas › Conversão | Regra única por fonte; sem equalização | Equalizar apagaria brilho e contraste, que são o objeto de estudo | Tomada (regra definida em F3 › Todas › Conversão) | D-007 |
| Plano › Todas › Aumento de dados | Sem aumento fotométrico no treino | As degradações controladas perderiam sentido | Tomada | D-008 |
| Plano › Todas › Negativos no treino | Subamostrar negativos, com correção de prior | Prevalência baixa no Plus | Tomada | D-009 |
| Plano › Todas › Limiar | Fixado na validação da fonte (sensibilidade 0,90) e reaplicado no externo | Reflete o uso real | Proposta | D-010 |
| Plano › Todas › Calibração | ECE com 15 faixas de igual tamanho como métrica principal | O Brier mistura calibração e discriminação | Tomada | D-011 |
| Plano › Todas › Nitidez e ruído | Comparados só dentro de cada base | Resolução, compressão e conversão diferem | Tomada | D-012 |
| Plano › Todas › Física | Parâmetros físicos só dentro do fabricante, sem converter em dose | Índices de exposição têm definições diferentes | Tomada | D-013 |
| Plano › Todas › Estatística | Holm nas primárias, Benjamini-Hochberg nas secundárias; margens 0,02 (AUROC), 0,02 (ECE), 0,03 (FNR/FPR) | Separar significância de relevância prática | Proposta | D-014 |
| Plano › Todas › Incerteza | Bootstrap por paciente (2.000) e dispersão entre sementes, reportados separados | São fontes de incerteza diferentes | Tomada | D-015 |
| Plano › Todas › Linguagem | A parcela da queda é "associada à padronização", não causal | Pesos de importância não identificam causa | Tomada | D-016 |
| Plano › Todas › Versões futuras | Multimodal na v2; pediatria na v3 | Prazo e foco | Tomada | D-017 |
| Plano › Todas › Espaço latente | Análise exploratória pré-especificada (fase 10) | Fase do roteiro sem cobertura no protocolo | Proposta | D-019 |
| Plano › Todas › Revisão dos erros | Revisão visual com livro de códigos (fase 9) | Sem pré-especificação, parece pesca de padrões | Proposta | D-020 |
| Plano › Todas › Atalhos cedo | T1 e T2 exploratórios em treino e validação | Mostram atalhos sem gastar o teste | Proposta | D-021 |
| Plano › Todas › Ensaio geral | Código de análise testado na validação e congelado antes de abrir o teste | Evita ajustes depois de ver o teste | Proposta | D-022 |
| Plano › Todas › Pré-registro | Tag no git e registro no OSF | Carimbo de tempo verificável | Proposta | D-023 |

---

## Fase 3 — decisões que valem para as três bases

| Decisão | Conteúdo | Por quê | Estado | Origem | Número antigo |
|---|---|---|---|---|---|
| F3 › Todas › Idade | Adultos de 18 a 100 anos, com idade registrada | Escopo de adultos; idades acima de 100 são erros de registro | Proposta | Auditorias finais | D-026 (Plus) |
| F3 › Todas › Primeira frontal | Uma frontal por estudo. Plus: menor número de vista. PadChest: PA, depois AP, depois AP deitado. RSNA: já uma por estudo | Não contar o mesmo rótulo duas vezes | Proposta | Auditorias finais | D-027 (Plus) |
| F3 › Todas › Duplicatas | Duplicatas exatas (arquivo ou pixels idênticos) entre pacientes diferentes: excluir todos os membros. No mesmo paciente: manter o estudo mais antigo | Vínculo imagem–laudo duvidoso | Proposta; **aplicada na passada completa (28 a 30/10)**: 1 par de pixels idênticos no RSNA e 1 no Plus, ambos do mesmo paciente; sai a imagem mais recente de cada par. Nenhuma duplicata quase idêntica (correlação ≥ 0,995) entre pacientes diferentes nas três bases | `chexpert_plus_final.md`, `rsna_final.md` | D-028 (Plus) |
| F3 › Todas › Sexo | Excluir sexo não registrado | É covariável da padronização | Proposta | Auditorias finais | D-029 (Plus) |
| F3 › Todas › Contagem | Positivos contados por imagem; intervalos por bootstrap de paciente; sensibilidade com peso igual por paciente | O mesmo paciente pode ter a doença em fases diferentes | Proposta | `rsna_final.md` | D-029 (RSNA) |
| F3 › Todas › Conversão | Mínimo-máximo por percentis 0,5–99,5 nas três bases, com inversão de MONOCHROME1. Janelas como sensibilidade, conforme a base: PadChest só no Philips (F3 › PadChest › Conversão); Plus como análise exploratória descritiva (F3 › Plus › Conversão) | O mínimo-máximo puro "lava" imagens com pixels extremos; regra única evita que a conversão acompanhe o fabricante | Proposta | `chexpert_plus_final.md`, `padchest_final.md` | D-030 (Plus) |
| F3 › Todas › Máscara | `lungs-segmentation` resnet34 (treinado só em Montgomery e Shenzhen), mesmo procedimento nas três bases; máscara central fixa como sensibilidade | Comparação justa; o CheXmask viu o PadChest e o TorchXRayVision viu o NIH | Proposta (teste nos positivos feito em 26/10: não depende da doença) | As três auditorias finais | D-036 (Plus), substitui D-031 (RSNA) |
| F3 › Todas › Redação | "Qualidade e apresentação nas imagens distribuídas", não "na aquisição" | Nenhuma base prova que a imagem não foi processada | Proposta | As três auditorias finais | D-035 (Plus) |
| F3 › Todas › Medidas de qualidade | As oito medidas da §6 calculadas na resolução original, sobre a imagem já convertida para 8 bits, dentro da máscara do pulmão (`src/qualidade/medidas.py`). Detalhes que o protocolo deixava em aberto: saturação = pixels ≤ 1 ou ≥ 254; enquadramento = fração da imagem em regiões pretas (≤ 5) ligadas à borda; amplitude = p95 − p5; entropia em bits (256 tons); nitidez = variância do Laplaciano 3 × 3; ruído = 1,4826 × MAD do resíduo (imagem − gaussiana σ = 1,5) nos 20% de pixels mais homogêneos da máscara. Auxiliares só descritivas (prefixo `q_aux_`): área da máscara e saturação nativa (pixels no mínimo ou máximo da imagem original) | Fixar as definições antes de olhar os dados; a saturação nativa recupera o teto do aparelho que a conversão por percentis esconde | Proposta | `src/qualidade/medidas.py` | — |
| F3 › Todas › Máscara × doença | A máscara **não depende da doença** pela regra fixada antes de rodar; segue o `lungs-segmentation`, sem lungVAE | Teste no RSNA em 26/10, 1.000 imagens (250 por projeção × rótulo). Cobertura relativa mediana 1,14 (limite 0,90). Área 4,5% menor nos positivos no AP e 4,7% no PA (limite 10%). Pulmão encolhido no AP: 5,2% contra 1,6% (3,6 pontos; limite 5). Nenhum dos três critérios atingido | Tomada | `rsna_final.md` §8.2; `src/imagem/teste_mascara_doenca.py` |
| F3 › Todas › Passada completa | Medidas de qualidade, duplicatas por pixel e resumo de resolução feitos na mesma passada que gera o cache de 320 px | As duas tarefas leem as mesmas imagens | Tomada; **feita**: RSNA 25.408 imagens (28/10), Plus 65.705 (28 a 30/10), PadChest 24.773 (segunda passada, 26 e 27/10) | `src/passada/passada_completa.py`, `rsna_final.md` §12, `chexpert_plus_final.md` §7 | D-037 (Plus) |
| F2 › Todas › Ambiente | Notebooks do Redivis, processamento em fluxo (baixar, processar, guardar o derivado, apagar o original), 8 downloads simultâneos | Disco de 60 GB por notebook; 54,4 MB/s com 8 downloads | Proposta | `chexpert_plus_final.md` | D-024 (Plus) |

---

## Fase 3 — CheXpert Plus

| Decisão | Conteúdo | Por quê | Estado | Número antigo |
|---|---|---|---|---|
| F3 › Plus › Rótulo | CheXbert da impressão do laudo. Positivo = 1; negativo explícito e não mencionado = 0; incerto excluído. Sensibilidades: incerto = 1; CheXbert dos achados | O arquivo do laudo inteiro está desalinhado com as imagens; a impressão está alinhada | Proposta | D-018 (revisada) |
| F3 › Plus › Exclusões | Excluir as imagens sem DICOM (10.981 frontais AP/PA); não usar os pacotes `.zip` | Os positivos não faltam mais que os demais (5,52% contra 5,75%) | Proposta | D-025 (Plus) |
| F3 › Plus › Recorte | Sem recorte de pacientes: base inteira | Mesmo inteira, rende só cerca de 689 positivos de teste | Proposta | D-033 |
| F3 › Plus › Faixas | 2 faixas de qualidade nas análises estratificadas | 689 positivos de teste: abaixo de 750, acima de 500 | Proposta | D-038 |
| F3 › Plus › Física | Parâmetros físicos só exploratórios, dentro da GE | Válidos em cerca de 20% da base, quase todos PA da GE | Proposta | D-031 (Plus) |
| F3 › Plus › Fabricante | Duas famílias (Agfa, GE), com o nome do fabricante unificado (`src/qualidade/fabricante.py`); declarar que posição e fabricante não se separam. As 14 imagens Varian ficam no estudo, mas fora das análises estratificadas por fabricante | AP é 99,3% Agfa; PA é 87,2% GE. Na passada: Agfa 56.575, GE 9.116, Varian 14; um subgrupo de 14 não sustenta análise própria | Tomada (30/10) | D-032 |
| F3 › Plus › Conversão | Percentis como conversão principal (é o que o modelo vê); medidas com a janela gravada no DICOM (GE e parte da Agfa) como **análise exploratória descritiva**, fora dos testes confirmatórios e da correção de Holm | Concordância percentis × janela baixa (brilho −0,04; demais medidas 0,30 a 0,68). A janela não corta a imagem (saturação mediana zero nas duas), só a apresenta de outro jeito, e é como o radiologista a via; as medidas já estão calculadas, então o custo é zero | Tomada (30/10) | — |
| F3 › Plus › Texto do protocolo | Contagens e prevalência passam a sair do CheXbert da impressão (1,95% / 2,13%); o F1 de 0,44 citado é do CheXbert-achados; o confundimento previsto (AP mais grave) não aparece | Os números do protocolo vêm de um rotulador que não está disponível | Proposta | D-034 |

---

## Fase 3 — RSNA

| Decisão | Conteúdo | Por quê | Estado | Número antigo |
|---|---|---|---|---|
| F3 › RSNA › Fonte | Distribuição oficial da RSNA, não o Kaggle | Termos compatíveis; mesmas imagens e rótulos | Proposta | — |
| F3 › RSNA › Rótulo | Grupo final "Calculated": positivo = Lung Opacity; negativo = Normal + Not Normal (classe guardada para sensibilidade) | É o rótulo final do desafio, uma classe por imagem | Proposta | — |
| F3 › RSNA › Exclusões | As 316 sem classe final, menores de 18 anos e idades acima de 100 | Fora do desafio; escopo; idades inválidas | Proposta | — |
| F3 › RSNA › Recorte | Treino do Kaggle na análise principal; as 3.000 do teste do Kaggle como sensibilidade | Teste enriquecido (36,5%) e com outro processo de rotulagem | Proposta | D-028 (RSNA) |
| F3 › RSNA › Paciente | Paciente = 8 primeiros dígitos do nome original no NIH | O `PatientID` do DICOM repete o ID da imagem | Proposta | — |
| F3 › RSNA › Física | Fora da análise com parâmetros físicos | Sem fabricante, janela nem exposição; JPEG com perda | Proposta | D-030 (RSNA, parte) |

---

## Fase 3 — PadChest

| Decisão | Conteúdo | Por quê | Estado | Número antigo |
|---|---|---|---|---|
| F3 › PadChest › Recorte | Teste externo = pacotes 0–11 e 54 (100% rotulados por médicos), frontais de adultos, uma por estudo, sem "exclude"; pneumonia atípica só na sensibilidade | Os pacotes são blocos de método (V de Cramér 0,99) e ano (0,57); rótulo médico; 796 positivos; critério fixado antes de ver imagens | Tomada | D-025 (PadChest) |
| F3 › PadChest › Termos | Dados sempre privados; publicar só resultados agregados e código; link de download não compartilhado | Cláusulas 5, 6 e 7 do acordo de uso | Proposta | — |
| F3 › PadChest › Exclusões | Fora dos pacotes escolhidos; idade fora de 18 a 100; não frontais; "exclude"; sexo "O" | Recorte e regras comuns | Proposta | — |
| F3 › PadChest › Escala | PNG reescalados para 16 bits; janela do CSV só vale depois da conversão | Medido nas 24.773 imagens | Proposta | — |
| F3 › PadChest › Segunda passada | Baixar de novo os 13 pacotes, um por vez, para calcular as medidas na resolução original e comparar as duas conversões (percentis e janela do CSV reescalada); guardar só as medidas e apagar cada pacote | As originais de 16 bits foram apagadas depois da conversão; o protocolo exige as medidas na resolução original e duas conversões no PadChest | Tomada; **feita em 26 e 27/10**: 24.773 imagens medidas, nenhuma máscara vazia | `padchest_final.md` §12 |
| F3 › PadChest › Máscara (caso #43) | O caso #43 é aceitável; a máscara do PadChest fica com 97 de 100 | A imagem mostra só um lado do tórax (recorte ou colimação); a máscara segue a anatomia visível | Tomada | — |
| F3 › PadChest › Fabricante | Análises de qualidade e física sempre dentro do fabricante | Saturação, fabricante e prevalência andam juntos; e, no Imaging Dynamics, as medidas dependem da conversão (segunda passada) | Proposta | `padchest_final.md` §12.1 |
| F3 › PadChest › Conversão | Percentis 0,5–99,5 como conversão principal; sensibilidade com a janela do CSV reescalada **só no Philips** | No Philips as duas conversões concordam (Spearman 0,89 a 0,99, exceto saturação); no Imaging Dynamics a janela corta as pontas (saturação no pulmão de 0% para 14%; ruído e entropia sem concordância) e não serve como verificação. Limitação conhecida, não escolha arbitrária | Proposta | `padchest_final.md` §12.1 |
| F3 › PadChest › Duplicatas quase idênticas | Nenhuma exclusão | Regra fixada antes: correlação ≥ 0,995 nas cópias de 320 px. 19.911 pares candidatos (hash igual), todos entre pacientes diferentes, nenhum acima de 0,98; inspeção visual dos 12 pares acima de 0,95: pessoas diferentes | Tomada | `padchest_final.md` §12.2 |


## F4 — Fechamento do protocolo

| Tema | Decisão | Por quê | Estado | Onde |
|---|---|---|---|---|
| F4 › Todas › Receita (razão de negativos) | 4 negativos por positivo no treino do Plus | É a razão usada na estimativa de GPU da F2 (10 a 20 h para os 21 treinos); uma razão maior exigiria nova passada no Plus | Tomada (28/10) | `configs/base.yaml` |
| F4 › Plus › Subamostra fixa | Os negativos do treino do Plus são sorteados uma vez (semente 2026); as 5 sementes de treino usam o mesmo conjunto e variam só a inicialização e a ordem dos lotes | Medir e guardar só o que o modelo usa; sortear de novo por semente exigiria medir bem mais imagens | Tomada (28/10) | `src/divisao/listas.py` |
| F4 › Plus › Passada | Passada completa do Plus só nas imagens que o estudo usa: teste e validação inteiros, treino com todos os positivos e os negativos sorteados; incertos do treino ficam fora (rótulo primário). Primeiro treino e validação, depois teste | Todas as 176.660 levariam cerca de 70 h de CPU; as análises confirmatórias usam só o teste, que é medido inteiro | Tomada (28/10) | `src/passada/passada_completa.py` |
| F4 › Todas › Listas | Divisão por paciente, 70/10/20, semente 2026, estratificada pelo rótulo do paciente (positivo, só incerto, negativo); PadChest inteiro como teste externo. Os IDs ficam só no Redivis; no repositório entram as contagens e o SHA-256 de cada lista | Termos de uso das bases; o hash prova que as listas não mudaram depois do congelamento | Tomada (28/10); **listas geradas em 28 a 30/10**: Plus 176.659, RSNA 25.402, PadChest 24.770 imagens. Positivos no teste: Plus 690, RSNA 1.100, PadChest 796 | `src/divisao/listas.py`, `data/splits/listas_sha256.json` |

---

## Decisões ainda abertas (fechar até o congelamento)

| Tema | O que falta decidir |
|---|---|
| F4 › Todas › Brilho e opacidade | O brilho dentro do pulmão é maior nos positivos com as duas máscaras, a normal e a preenchida (diferença padronizada 0,39 no AP e 0,16 no PA, teste da máscara de 26/10). Parte do "brilho" é a própria opacidade, não a apresentação da imagem. Decidir como o eixo A e o T1 leem essa medida |
| F4 › Todas › Receita | Resolução em que as degradações do eixo B são aplicadas (a razão de negativos já foi decidida) |
| F4 › Todas › Testes primários | Lista nominal da família corrigida por Holm |
