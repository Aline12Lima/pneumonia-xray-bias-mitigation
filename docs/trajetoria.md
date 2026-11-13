# Trajetória do projeto

Como o projeto chegou até aqui, passo a passo: o que foi feito em cada etapa, o que se descobriu e que decisão isso gerou. Os detalhes estão nos documentos indicados em cada passo; as decisões estão em `decisoes.md`.

---

## 1. Estudo anterior (janeiro de 2026)

**Documento:** `research/previous_study/README.md`

- **O que foi feito:** DenseNet121 com e sem módulo de atenção (CBAM), treinada na base pediátrica Kermany, com análise do espaço latente (UMAP e HDBSCAN).
- **Conclusão da época:** o modelo sem atenção agrupava as imagens por brilho; com atenção, por morfologia. A hipótese era "viés de luminosidade".
- **Limites que motivaram o novo estudo:** uma só base, pediátrica e de uma só instituição; sem teste externo; "brilho" tratado como causa sem saber de onde ele vinha.

## 2. Fundamentação física e clínica (6/10/2026)

**Documentos:** `fundamentacao/1A` a `1F`

- **O que foi feito:** estudo da cadeia que forma a radiografia: raios X, exposição, detector, geometria, pós-processamento, conversão e qualidade clínica.
- **O que se descobriu:** o brilho da imagem final não mede a exposição. Ele resulta de aquisição, processamento do fabricante, janela e conversão. Várias condições reais (AP/PA, inspiração, rotação do paciente) não podem ser simuladas numa imagem pronta.
- **O que mudou:** a hipótese "luminosidade causa viés" virou "características de qualidade e apresentação se associam ao erro e à generalização". Cada fator recebeu um tratamento: medir, perturbar, estratificar ou declarar como limitação (matriz do `1F`).

## 3. Escolha das bases (8/10/2026)

**Documento:** `escolha_das_bases.md`

- **O que foi feito:** levantamento documental de cinco famílias de bases: RSNA, CheXpert/CheXpert Plus, PadChest, PediCXR e Kermany.
- **O que se descobriu:** os rótulos têm significados diferentes em cada base (opacidade vista na imagem, menção no laudo, rótulo de médico). DICOM nem sempre quer dizer imagem crua.
- **Decisão:** selecionar as cinco e auditar a viabilidade antes de fixar papéis. Depois, no protocolo, a pediatria saiu da v1 (Plano › Todas › Versões futuras).

## 4. Auditoria preliminar (13 a 17/10/2026)

**Documentos:** `auditorias/rsna_preliminar.md`, `chexpert_plus_preliminar.md`, `padchest_preliminar.md`

| Base | O que se fez | O que se descobriu | Decisão gerada |
|---|---|---|---|
| RSNA (13/10, Kaggle) | Integridade, rótulos, caixas, cabeçalhos, pixels de amostra | Arquivos íntegros; "Target = 0" mistura normal e outras alterações; **projeção fortemente associada ao rótulo** (38% de positivos no AP contra 9% no PA); ID de paciente do Kaggle não é paciente | Aprovada com restrições; mapear o paciente pelo NIH |
| CheXpert Plus (14/10, Redivis) | Cobertura de caminhos, DICOM, rótulos por seção do laudo | 12.493 imagens sem DICOM solto; **grande divergência entre os rótulos das seções do laudo** | Manter como candidata; definir a fonte de rótulo antes de usar |
| PadChest (17/10, só metadados) | CSV, listagens dos 52 pacotes, campos DICOM | **Os pacotes não são aleatórios** (blocos de método de rotulagem e de ano); sem kVp; cada fabricante registra campos diferentes | **F3 › PadChest › Recorte:** pacotes 0–11 e 54, rotulados por médicos |

**Protocolo v1.0 e plano de execução (17/10):** escritos com base nessa rodada (`protocolo/`). Definiram as três bases, os três modelos, os eixos de análise e os portões.

## 5. Auditorias finais (18 a 23/10/2026)

**Documentos:** `auditorias/rsna_final.md`, `chexpert_plus_final.md`, `padchest_final.md`; síntese em `02_auditoria_dos_dados.md`

### CheXpert Plus (19 a 21/10)

| Passo | Achado | Decisão |
|---|---|---|
| Rótulos disponíveis | São do CheXbert, não do rotulador CheXpert citado no protocolo | F3 › Plus › Texto do protocolo |
| Teste de alinhamento dos rótulos | **O arquivo do laudo inteiro está desalinhado com as imagens**; impressão e achados estão alinhados | F3 › Plus › Rótulo (impressão) |
| Tamanho do DICOM | 2,58 TB; a base inteira rende só cerca de 689 positivos de teste | F3 › Plus › Recorte (sem recorte) e F3 › Plus › Faixas (2 faixas) |
| Cabeçalhos (2.000) | AP quase todo Agfa, PA quase todo GE; parâmetros físicos em cerca de 20% | F3 › Plus › Fabricante e F3 › Plus › Física |
| Conversão | Mínimo-máximo puro "lava" imagens | F3 › Todas › Conversão (percentis) |
| Máscara (100 imagens) | 95/100, no limite | F3 › Todas › Máscara |
| Composição | **Prevalência maior no PA (3,19%) que no AP (1,72%)**: o contrário do RSNA | Reforça o teste de atalhos |

### RSNA (18 e 21/10)

| Passo | Achado | Decisão |
|---|---|---|
| Fonte | A distribuição oficial tem as mesmas imagens, termos mais claros e o mapeamento de paciente | F3 › RSNA › Fonte |
| Rótulos oficiais | O grupo "Calculated" é o rótulo final; o teste do Kaggle é enriquecido (36,5%) | F3 › RSNA › Rótulo e F3 › RSNA › Recorte |
| Pixel | Processado duas vezes (PNG de 8 bits, depois JPEG com perda); sem fabricante nem exposição | F3 › RSNA › Física |
| Máscara | O CheXmask viu o PadChest no treino e não serve para as três bases | F3 › Todas › Máscara (mesmo modelo nas três); 100/100 |

### PadChest (22 e 23/10)

| Passo | Achado | Decisão |
|---|---|---|
| Termos de uso | Redistribuição de qualquer parte proibida | F3 › PadChest › Termos |
| Abertura das 24.773 imagens | Todas abrem; PNG reescalados para 16 bits | F3 › PadChest › Escala |
| Saturação | **O Imaging Dynamics satura o branco em 59% das imagens e tem um terço da prevalência do Philips** | F3 › PadChest › Fabricante |
| Máscara | 97/100; falhas em pulmões muito opacos | Mesmo padrão nas três bases: testar a máscara nos positivos |

### Resultado da fase

**As três bases foram aprovadas para entrar no estudo** (Plus com ressalvas). Três achados atravessam as bases e pesam na análise:

1. **A projeção se associa à pneumonia em direções opostas** entre bases (RSNA e PadChest: mais no AP; Plus: mais no PA). É uma explicação testável para a queda externa.
2. **Fabricante, qualidade e prevalência andam juntos** (Plus e PadChest).
3. **Nenhuma base traz a imagem crua.** Por isso a redação passa a ser "imagens distribuídas".

## 6. Problema e hipótese refinados (23/10/2026)

**Documento:** `01_problema_e_hipotese.md`

A auditoria não mudou a pergunta, mas mudou o peso das partes: parâmetros físicos viraram exploratórios, os controles de rótulo passaram a ser centrais, e a hipótese principal ficou testável (padronizar a qualidade reduz a queda de AUROC além da padronização clínica).

## 7. Organização da documentação (23/10/2026)

Os documentos produzidos em vários chats foram reunidos numa só pasta (`docs/`), sem repetição: no máximo uma auditoria preliminar e uma final por base, um registro único de decisões e esta trajetória. Diários, versões intermediárias e decisões soltas foram absorvidos aqui.

---

## 8. Passadas completas, listas e congelamento do protocolo (28 a 30/10/2026)

**Documentos:** `protocolo/protocolo_v1.1.md` e `.pdf`, `decisoes.md` (seção F4), `../data/splits/listas_sha256.json`

- **Passadas completas** nas três bases: as oito medidas na resolução original, a máscara, as duplicatas e o cache de 320 px (RSNA 25.408 imagens; Plus 65.705, só as que o estudo usa; PadChest 24.773). Uma duplicata de pixels saiu do RSNA e uma do Plus, as duas do mesmo paciente.
- **Listas** de treino, validação e teste por paciente (semente 2026), com SHA-256. Positivos no teste: Plus 690 (2 faixas de qualidade), RSNA 1.100 e PadChest 796.
- **Decisões do fechamento (30/10):**
  - brilho e opacidade: regras de leitura do brilho;
  - CheXthought: fora da v1.1;
  - degradações do eixo B: opção C, com controle positivo em 320 px, depois de um piloto em 50 imagens da validação do Plus e do RSNA. O piloto mostrou que o ruído e o desfoque leve na resolução original quase não chegam a 320 px;
  - margens: o ECE escalado pela prevalência, mais a regra das três saídas;
  - três famílias de testes primários com Holm (72, 27 e 5 testes).
- **Congelamento do protocolo v1.1 (30/10/2026)**, um dia antes da data do plano, a pedido da autora:

| Item | Valor |
|---|---|
| Arquivo | `docs/protocolo/protocolo_v1.1.pdf` (9 páginas) |
| SHA-256 do PDF | `97d4ee405c4c86f889e3b800ee4f0d6f0b3a1fb4953a91d2f707562de721835f` |
| Hashes das listas | `data/splits/listas_sha256.json` (teste: Plus `7437a920…`, RSNA `c4de2e1c…`, PadChest `0786ca66…`) |
| Trava do teste | Nenhum código de treino ou de análise lê `listas_teste_trancado/`; só `src/divisao/listas.py`, que a grava (verificado por busca no repositório em 30/10). O teste de fumaça (`src/fumaca.py`) usa só imagens sintéticas |
| Tag no git | `protocolo-v1.1` |
| Decisões | Todas tomadas; as que eram propostas viraram "Tomada no congelamento (30/10)" |

**A partir daqui, qualquer mudança no que está no protocolo v1.1 é um desvio**, registrado em `desvios/` com data, seção, motivo e impacto.

## 9. Linha de base (F5, a partir de 30/10/2026)

- **Passo 5.1 (30/10):** SHA-256 de cada arquivo do cache de 320 px, conferido contra as listas congeladas. Nenhum arquivo faltando: Plus 12.100 (treino sorteado), 18.147 (validação) e 35.457 (teste); RSNA 17.948, 2.417 e 5.037. Os hashes por partição estão em `../data/splits/cache_sha256.json`. O cache oficial do PadChest (8 bits, percentis na resolução original) é gerado na leitura dos pacotes da F8.
- **Passo 5.2:** carregador do cache (`src/treino/carregador.py`), que confere os hashes antes de usar as imagens e não lê o teste.
- **Passos 5.3 e 5.4 (1/11), piloto do M1, semente 0:** notebook "treino" do workflow `treino_modelos`, GPU NVIDIA L4 (g2-standard-4, Spot), cerca de 119 s por época (treino de 12.100 imagens e validação de 16.599). As duas taxas da grade pararam sozinhas pela AUPRC (paciência 5):

| Taxa | Melhor época / épocas rodadas | AUROC | AUPRC (prevalência 0,020) | ECE sem → com correção de prior | Probabilidade média corrigida |
|---|---|---|---|---|---|
| 1e-4 | 7 / 13 | 0,779 | 0,109 | 0,083 → 0,032 | 0,034 |
| 3e-4 | 9 / 15 | 0,768 | 0,099 | 0,082 → 0,018 | 0,022 |

  A correção de prior (π natural 0,0215; π do treino 0,20) leva a probabilidade média para perto da prevalência observada. A escolha da taxa é feita na F6 (passo 6.1).
- **Passo 5.5 (1 e 2/11), sanidade** (`src/treino/sanidade.py`), com critérios fixados no código antes de rodar:
  - **pacientes:** nenhum paciente no treino e na validação ao mesmo tempo (Plus e RSNA), as quatro listas iguais às congeladas pelo SHA-256 e nenhum arquivo de teste em `/out/dados`: aprovado;
  - **sobreajuste de um lote:** 32 imagens do Plus (16 positivas), sem aumento, 200 passos: perda 0,718 → 0,0004 (critério < 0,01): aprovado;
  - **rótulos embaralhados:** M1 com os rótulos do treino permutados, avaliado na validação com os rótulos reais. Pelo critério original (|AUROC − 0,5| < 0,05 na melhor época), a semente 0 foi **reprovada** (0,5552, melhor época 0). Escolher a melhor de várias épocas sem sinal puxa o valor para longe de 0,5, então o critério foi revisto **antes** de rodar novas permutações: AUROC média de todas as épocas, |média − 0,5| < 0,05, em três permutações. Resultado: semente 0 média 0,510 (6 épocas), semente 1 0,492 (9), semente 2 0,503 (16), todas aprovadas; as melhores épocas teriam dado 0,555, 0,573 e 0,540. Os arquivos ficam em `/out/sanidade/` (`embaralhado.json` guarda a reprovação pelo critério original).
  - **Correções feitas no caminho:** o `--retomar` treinava uma época a mais depois da parada antecipada e falhava na GPU ao restaurar o gerador aleatório; as duas falhas apareceram em quedas da Spot e foram corrigidas antes de qualquer treino da F6.
- **Passo 5.6 (3/11), determinismo:** repetição do piloto (M1, semente 0, taxa 1e-4), já com todas as correções do código: predições **idênticas bit a bit** (maior diferença de logit 0; ΔAUROC 0 nas 13 épocas; mesma melhor época, 7). Tolerância registrada antes de rodar: \|ΔAUROC\| < 0,01. Aprovado.
- **Passo 5.7 (3/11), tabela de predições:** covariáveis enviadas ao `cache_treino` (RSNA 20.365 imagens, idade 18 a 92, sem fabricante; Plus 30.247, idade 18 a 89, Agfa 26.074, GE 4.164, Varian 9; uma imagem do Plus com máscara vazia, incerta). Tabela da validação do M1 (semente 0, taxa 1e-4): 16.599 linhas, 5.999 pacientes, 334 positivos, 26 colunas (18 do schema e 8 medidas), probabilidade média 0,0343. Válida contra o schema.
- **Passo 5.8 (3/11), orçamento:** ver F5 › Todas › Orçamento. 5 sementes mantidas; 16 treinos restantes, cerca de 78 créditos estimados contra 72,58 disponíveis; compra de 50 ou 100 créditos se faltar.
- **Portão G5 (linha de base estável), aprovado em 3/11:** M1 estável na validação (piloto nas duas taxas, sanidade aprovada); diferença entre execuções idênticas zero, abaixo da tolerância; orçamento decidido, com a compra de créditos como reserva.
- **Máquina e créditos:** a Spot caiu várias vezes e, na manhã de 2/11, não havia L4 livre nem na standard (7,55 créditos por hora, contra cerca de 4,6 na Spot). Saldo: 91,58 créditos antes do piloto, 75,21 depois da sanidade. A saída da célula do notebook às vezes para de atualizar com o treino ainda rodando; o progresso real fica no `curvas.csv`.

## 10. Modelos (F6, a partir de 3/11/2026)

- **Passo 6.1, busca de taxa:** M1 pelos pilotos da F5 (1e-4); M2 em 3/11, semente 0: 1e-4 → AUROC 0,874, AUPRC 0,643 (prevalência 0,199), melhor época 2 de 8; 3e-4 → AUROC 0,872, AUPRC 0,648, melhor época 5 de 11. Escolhida 3e-4 (F6 › M2 › Taxa). No RSNA a correção de prior é zero (sem subamostragem), como esperado. A tela do notebook parou de atualizar durante os dois treinos; eles terminaram normalmente e a retomada só refez a avaliação final. A partir daqui, cada treino grava também um `log.txt` na pasta da execução.
- **Passo 6.1, M3 (3/11):** 1e-4 → média das fontes AUROC 0,826, AUPRC 0,377 (Plus 0,782 / 0,099; RSNA 0,869 / 0,655), melhor época 3 de 9; 3e-4 → 0,827 / 0,375, melhor época 9 de 15. Escolhida 1e-4. Correção principal pela regra: 3c nas duas (ECE médio 0,016 e 0,027, contra 0,053 e 0,052 da 3a).
- **Passo 6.2 (4 a 7/11), sementes 0 a 4 dos três modelos, todas na L4:**

| Modelo (taxa) | AUROC, média (DP) entre as 5 sementes | AUPRC, média (DP) | Épocas até a parada |
|---|---|---|---|
| M1 (1e-4), validação do Plus | 0,785 (0,010) | 0,112 (0,003) | 8 a 13 |
| M2 (3e-4), validação do RSNA | 0,874 (0,003) | 0,648 (0,010) | 8 a 13 |
| M3 (1e-4), Plus / RSNA | 0,784 (0,010) / 0,872 (0,007) | 0,099 (0,008) / 0,659 (0,008) | 8 a 10 |

  No M3, a correção 3c venceu em todas as sementes. O M3 empata com o especialista de cada fonte. Muitas quedas da Spot e um dia sem L4 disponível (5/11); o `--retomar` e o `log.txt` evitaram perda de treino.
- **Passos 6.3 e 6.4 (7/11), limiar e temperatura:** sensibilidade 0,90 na validação da fonte em todos os 15 modelos. FPR no limiar: M1 0,58 a 0,63; M2 0,33 a 0,37; M3 (conjunta) 0,36 a 0,42. Temperatura de 1,08 a 1,69 (modelos um pouco confiantes demais). ECE depois da temperatura às vezes melhora, às vezes piora (a temperatura minimiza a log-verossimilhança, não o ECE). No M3, o limiar conjunto dá sensibilidade de cerca de 0,76 no Plus e 0,996 no RSNA: ver F6 › M3 › Limiar por fonte.
- **Passo 6.5a (7/11), predições do ensaio geral:** os 15 modelos aplicados à validação do Plus e à do RSNA (285.240 linhas, schema sem erros). Na validação da própria fonte, o logit bate com o do treino até 1 a 4 passos de fp16 (2⁻⁸ a 2⁻⁶), diferença do arredondamento da GPU; a tolerância da conferência foi ajustada para 0,02.
- **Passo 6.5, eixo B (8/11):** modelos levados aos notebooks das bases (dataset `modelos`, SHA-256 conferido). Amostra do ensaio: 100 positivos e 400 negativos da validação por base; 29 níveis (gama p1/p99 e brilho p1 fora da grade no Plus; gama p95/p99 e brilho p99 no RSNA); 15.000 versões por base, 225.000 predições por base. A ancoragem do piloto não incluía a imagem natural na grade (corrigido). Erro de ancoragem: fotométricas e ruído perto do alvo (brilho com desvio constante de cerca de 7 tons, da amostra); **desfoque leve e resolução muito longe do alvo** (a nitidez cai de forma não linear; qualquer reamostragem já derruba a nitidez), o que exige grade fina perto de zero na F8.
- **Passo 6.5, análises do ensaio geral (8/11, validação, bootstrap 200):** E0, eixo A, T1 e T2, eixo C, eixo B e consolidação rodaram de ponta a ponta sobre as predições da validação (o cruzamento Plus ↔ RSNA no papel do externo). Números de depuração, não resultados. O que o ensaio mudou no código antes do teste:
  - **eixo A:** separação perfeita em alguns estratos (estouro no ajuste) → passo reduzido na regressão e regra de estimabilidade (≥ 10 eventos por parâmetro, EP ≤ 2,5, |coef| ≤ 10; falha entra com p = 1). A biblioteca `statsmodels` do Redivis é incompatível com o pandas instalado: a regressão com EP agrupado foi implementada em numpy e conferida contra o `statsmodels` fora do Redivis;
  - **eixo C:** a diferença padronizada (SMD) passou a ser medida dentro de cada classe; o S2 não fica balanceado (T2 com AUROC de cerca de 0,99 só com as medidas comparáveis; suporte comum de 12% a 16%), então foi acrescentada a regra de leitura (SMD ≥ 0,1 → parcela inconclusiva) e o suporte comum (0,1 a 0,9). Expectativa: H2 provavelmente inconclusiva no teste;
  - **eixo B:** a mesma regra de estimabilidade valeu para FNR e FPR (um p de 7e-238 vinha de separação); FNR do M3 no RSNA com poucos eventos na amostra do ensaio;
  - **família 2:** 18 testes (gama, brilho e contraste × 6 pares do ensaio); nenhum passa no Holm com 200 reamostragens (menor p possível 0,005). **No teste, usar 2.000 reamostragens** (o padrão do código);
  - **T3:** a comparação de direção usava efeitos nulos do eixo A; a regra foi revista (só onde A e B têm efeito; brilho e gama só no FP) antes do teste. Resultado do ensaio: 5 de 48 curvas;
  - **respostas em U:** brilho e gama aumentam o FPR nos dois lados (clarear e escurecer), o que justifica os dois lados por degradação.
- **Lições para a F8:** grade de ancoragem fina perto de zero para desfoque e resolução; controle negativo (p25/p75) fora da margem em várias curvas do RSNA, a conferir no teste com mais imagens.
- **Incidente (8/11), PadChest:** o notebook não ligava porque o dataset de origem tinha sido excluído (falta de espaço). Ao remover esse dataset do workflow `auditoria_padchest`, o Redivis apagou também o notebook `00_padchest_auditoria` e o seu `/out`, sem aviso. Pedido de restauração enviado ao suporte. Em paralelo, reconstrução num workflow novo (F6 › PadChest › Reconstrução); o CSV original está no próprio link da BIMCV. Lição: antes de remover um item de um workflow, guardar fora do notebook (num dataset privado) tudo que estiver só no `/out`.
- **Reconstrução do PadChest concluída (9/11):** (1) recorte refeito do CSV da BIMCV com a regra do Bloco P0: 24.773 imagens, 22.987 pacientes, 796 positivos; (2) lista refeita com **SHA-256 idêntico ao congelado** (`0786ca66…`, 24.770 imagens); (3) segunda passada com `--com-cache` nos 13 pacotes (3 sessões, cerca de 1,2 s por imagem): as 24.773 imagens abriram, nenhuma máscara vazia, 24.601 com janela no CSV, e as tabelas de concordância por fabricante e as 4.565 candidatas a duplicata em 1.364 grupos **iguais às da auditoria de 27/10** (seção 12 do `padchest_final.md`); (4) cache de 320 px com as 24.770 imagens da lista, SHA-256 do conjunto `7fc2f81d…` (em `data/splits/cache_sha256.json`). Nada mudou nas regras nem nos dados: incidente fechado, sem desvio.
- **Antes da tag (6.6), caminho do teste:** cache de 320 px do PadChest (F6 › PadChest › Cache de 320 px), predição do teste com trava até o destrancamento e "modo simulado" na validação.
- **Caminho do teste (9/11):** `src/predicoes/teste.py` com trava (F6 › Teste › Trava), modo simulado (validação no Plus e no RSNA; PadChest sem modelo) e tabela à parte dos incertos do Plus. Roda no notebook de cada base, com GPU; as tabelas vão ao dataset `resultados`.
- **Modo simulado do caminho do teste (9 a 11/11):** PadChest sem modelo (24.770 imagens, 796 positivos, 22.985 pacientes; hashes do cache conferidos; covariáveis e schema OK); RSNA na validação (15 execuções, 2.417 imagens, 480 positivos); Plus na validação (15 execuções, 16.599 imagens + 1.548 incertas na tabela à parte, 334 positivos). Probabilidades médias coerentes com o ensaio (M2 no Plus perto de 0,2: o M2 foi treinado com a prevalência natural do RSNA, sem subamostragem).
- **Análises com três fontes (11/11):** no ensaio, o PadChest não tinha validação e o caminho "M3 → externo" nunca tinha rodado. Uma tabela sintética com as três fontes (15 execuções, schema válido) passou por E0, eixo A, T1/T2 (T2 com 3 classes) e eixo C sem erro; os pares M1/M2/M3 → PadChest aparecem como esperado. `src/predicoes/juntar_teste.py` junta as tabelas das três bases no notebook de treino e confere cada uma contra as contagens congeladas.
- **Eixo B no teste (11/11):** grade fina para a ancoragem da F8 (F6 › Eixo B › Ancoragem da F8), ancoragem do PadChest sem rótulo e sem modelo, e geração e predição do teste com a mesma trava (F6 › Eixo B › Teste). Falta rodar a ancoragem da F8 nas três bases antes da tag.
- **Ancoragem da F8, primeira rodada (12/11):** desfoque e resolução passaram a ter níveis intermediários nas três bases (lição do ensaio resolvida). Faltaram gama p99 no RSNA e gama/brilho p1 no Plus → grade estendida (gama até 3,0; brilho ±80). PadChest completo (todos os 32 níveis), pacote 0, 50 imagens. Resolução sem dose gradual no Plus e nos dois primeiros níveis do PadChest (F6 › Eixo B › Resolução sem dose gradual). Ruído: no original os níveis são alcançados, mas em 320 px sobra só 4 a 10% do efeito.
- **Ancoragem da F8 aprovada (12/11):** com a grade final (gama 0,4 a 3,0; brilho ±80), os 32 níveis têm parâmetro nas três bases (Plus e RSNA refeitos; PadChest completo já na primeira rodada, sem precisar refazer). Exemplos: gama p1 no Plus 2,313; brilho p1 no Plus −72,2; gama p99 no RSNA 0,480.
- **Antes da tag, peças pré-especificadas que faltavam (12/11):** sensibilidade da conversão do PadChest nas predições (F6 › PadChest › Sensibilidade da conversão) e amostra e livro de códigos da revisão qualitativa (F9 › Revisão qualitativa). Falta o código exploratório da F10.
- **F10 escrita antes da tag (12/11):** embeddings, sondas de origem e de qualidade, projeção e Grad-CAM com sanidade (F10 › Espaço latente e explicabilidade › Desenho), testados com dados sintéticos. Com isso, todo o código de análise pré-especificado está no repositório: próximo passo, a tag (6.6).
- **Passo 6.6, tag do código (13/11):** a primeira tentativa (10h02) criou `analise-v1` em `0c39248`, sem a F10, porque o patch da F10 não tinha sido baixado; a tag foi apagada (local e GitHub) antes de qualquer leitura do teste e recriada às 10h06 em `d71aae1`, com todo o código pré-especificado (predições do teste com trava, junção, E0, eixos A, B e C, T1/T2, T3, sensibilidade da conversão, F9 e F10).
- **Passo 6.7, destrancamento (2026-11-13):** `docs/destrancamento.json` registra a hora, a tag `analise-v1` (`d71aae1`) e o SHA-256 das três listas de teste (Plus `7437a920…`, RSNA `c4de2e1c…`, PadChest `0786ca66…`). O teste fica destrancado a partir do merge desse arquivo; o portão G6 (15 modelos, métricas de validação registradas, tag antes de qualquer leitura do teste) está cumprido.
- **Créditos (7/11):** o Redivis concedeu 500 créditos; saldo 518.
- **M3 implementado (3/11):** época balanceada, parada pela média das AUPRCs das fontes e correções 3a e 3c com escolha pela menor ECE médio (decisões F6 › M3). O caminho do M1 e do M2 ficou idêntico: um treino curto na CPU deu predições iguais bit a bit antes e depois da mudança.

---

## Próximos passos (pelo plano de execução)

1. **Linha de base (1 a 7/11):** piloto do M1 com GPU, testes de sanidade e de determinismo, primeira tabela de predições da validação.
2. **Em paralelo:** T1 e T2 exploratórios em treino e validação; rótulos do rotulador CheXpert no Plus; geração das imagens degradadas da amostra (CPU).
3. **Modelos M1 a M3 (8 a 20/11)**, ensaio geral na validação, tag do código de análise e destrancamento do teste em 21/11.
