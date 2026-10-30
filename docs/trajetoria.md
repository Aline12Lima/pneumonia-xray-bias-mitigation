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

---

## Próximos passos (pelo plano de execução)

1. **Linha de base (1 a 7/11):** piloto do M1 com GPU, testes de sanidade e de determinismo, primeira tabela de predições da validação.
2. **Em paralelo:** T1 e T2 exploratórios em treino e validação; rótulos do rotulador CheXpert no Plus; geração das imagens degradadas da amostra (CPU).
3. **Modelos M1 a M3 (8 a 20/11)**, ensaio geral na validação, tag do código de análise e destrancamento do teste em 21/11.
