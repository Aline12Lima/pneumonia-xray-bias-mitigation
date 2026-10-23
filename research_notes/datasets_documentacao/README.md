
## Fontes


## 1. Objetivo científico preservado

> Como características de qualidade e apresentação das radiografias de tórax se relacionam com os erros, a calibração e a generalização de modelos de classificação entre diferentes bases de dados?

A aplicação permanece a **detecção de pneumonia em radiografias de tórax**. A investigação contempla luminosidade, outras características de qualidade e apresentação, falsos positivos, falsos negativos, confiança das previsões e avaliações externas.

A inclusão de bases infantis amplia as possibilidades de investigação. O projeto não foi delimitado como exclusivamente pediátrico ou exclusivamente adulto. A cobertura efetiva das populações e as comparações viáveis serão determinadas a partir dos dados.

Estão previstas diferentes modelagens e avaliações, orientadas por perguntas e hipóteses explícitas. A DenseNet121 permanece como baseline previsto no cronograma, sem limitar o estudo a um único modelo. As escolhas experimentais serão registradas no protocolo antes dos experimentos principais.

**A proposta discutida de substituir pneumonia por consolidação não foi adotada.** Eventuais achados associados poderão ser estudados futuramente, sem mudar automaticamente o objetivo aprovado.

## 2. Decisão final da fase 2

| Decisão da fase 2 | Definição |
|---|---|
| Três famílias inicialmente selecionadas | RSNA, CheXpert/CheXpert Plus e PadChest |
| Inclusões pediátricas | PediCXR e Kermany |
| Compatibilidade identificada | Aplicação relacionada à pneumonia em radiografias de tórax, com diferenças de referência que deverão orientar as comparações |
| O que ainda não está decidido | Subconjuntos, correspondência operacional dos rótulos, papéis A/B/C e experimentos finais |
| Próxima fase | Auditar a viabilidade real dos dados |

São **cinco famílias selecionadas**, com seis fichas individuais porque CheXpert original e CheXpert Plus foram documentados separadamente. O Plus é uma extensão da família Stanford, não um domínio hospitalar independente do original. PediCXR e VinDr-PCXR são nomes do mesmo recurso considerado neste estudo. [3–6, 9–10]

Nenhuma das cinco famílias foi descartada. A seleção não obriga a utilizar todas no mesmo treinamento ou em uma única comparação. As contribuições específicas serão delimitadas após a auditoria.

## 3. Síntese dos estudos dos dados

As dimensões são referências documentais: imagem, estudo e paciente são unidades diferentes. Versões, exclusões e subconjuntos podem alterar as contagens disponíveis.

| Família | Origem e população documentadas | Conteúdo e referência | Contribuição potencial para o DIAMANTE |
|---|---|---|---|
| **RSNA Pneumonia Detection Challenge (2018)** | NIH Clinical Center, EUA; inclui adultos e menores. Coleção de 30.000 exames e 12.274 pacientes únicos descrita no artigo. | Frontais AP/PA; PNG convertido para DICOM; anotações visuais de opacidades suspeitas de pneumonia, incluindo localização. | Investigar erro em relação à referência especializada e às características das imagens. [1–2] |
| **CheXpert / CheXpert Plus** | Stanford, EUA; família predominantemente adulta. Original: 224.316 radiografias de 65.240 pacientes. | Original em JPEG, com rótulos de treino extraídos de laudos; referências de avaliação por especialistas. Plus acrescenta DICOM, laudos e metadados. | Investigar apresentação, referência textual, incerteza e aquisição quando houver metadados utilizáveis. [3–6] |
| **PadChest** | Hospital San Juan, Alicante, Espanha. Faixa descrita de 0–105 anos, com predomínio de idades adultas. Conjunto final da tabela 7: 160.868 imagens de 67.625 pacientes. | PNG; diferentes projeções; metadados de aquisição; rótulos derivados de laudos por anotação manual ou automática. | Estudar outra origem hospitalar, projeções, informações técnicas e procedência dos rótulos. [7–8] |
| **PediCXR / VinDr-PCXR** | Vietnã; fonte original descreve menores de dez anos. 9.125 imagens/estudos; intervalo real de idade ainda deve ser conferido. | DICOM; projeção descrita como PA; anotações visuais de achados e doenças, incluindo caixas de localização. | Incorporar população infantil e verificar possibilidades de análise por idade e de generalização. [9–10] |
| **Kermany / Guangzhou Chest X-Ray** | Guangzhou Women and Children’s Medical Center, China; crianças de 1–5 anos. Referência de 5.856 imagens na literatura consultada. | Radiografias AP em JPEG; classes normal, pneumonia bacteriana e viral. | Acrescentar outra origem pediátrica e investigar transferência entre coleções infantis. [11–13] |

AP = anteroposterior; PA = posteroanterior. A disponibilidade de uma projeção ou idade na descrição não garante quantidade suficiente para qualquer análise de subgrupo.

### Laudos, rótulos e dados clínicos

- **RSNA:** a referência final resulta de anotação visual especializada. Laudos completos não são descritos como componente do pacote estudado. [1–2]
- **CheXpert original:** o treinamento utiliza rótulos extraídos dos laudos, sem que isso signifique disponibilização dos laudos completos na versão original. **CheXpert Plus** oferece texto associado às imagens. [3–6]
- **PadChest:** parte dos laudos foi anotada manualmente e parte automaticamente; o campo textual distribuído é descrito como trecho processado da interpretação radiográfica. [7–8]
- **PediCXR e Kermany:** a documentação consultada não estabelece uma distribuição equivalente de laudos completos para nosso uso. [9–13]

A existência de histórico clínico em alguns laudos abre uma possibilidade futura de estudo de sintomas. Isso permanece uma frente a avaliar: histórico, achados e conclusão precisam ser distinguidos, inclusive para evitar que a entrada do modelo contenha a resposta usada como rótulo.

### Aquisição e apresentação

Não foi demonstrado que as bases tenham equipamentos, detectores, exposição ou processamento equivalentes. DICOM, por si só, não garante imagem bruta do detector ou todos os parâmetros de aquisição. A conversão PNG→DICOM da RSNA é um exemplo documentado da importância de conhecer a procedência dos arquivos. [1]

CheXpert Plus e PadChest descrevem campos técnicos que podem apoiar a investigação. O preenchimento, as unidades e a utilidade desses campos serão verificados na fase 3. [5–8]

## 4. O que significa a compatibilidade identificada

A compatibilidade estabelecida nesta fase é **preliminar e documental**: as coleções permitem investigar uma aplicação relacionada à pneumonia em radiografias de tórax e oferecem diferenças relevantes para o estudo de qualidade, erro e generalização.

Não foi estabelecida equivalência universal entre os padrões de referência. Por exemplo:

- A RSNA marca opacidades consideradas suspeitas de pneumonia. [2]
- CheXpert distingue rótulos positivos, negativos, incertos e sem menção. [4]
- PadChest distingue achados radiográficos e diagnósticos derivados de laudos. [7–8]
- Uma comparação com negativos apenas normais responde a uma pergunta diferente de outra que inclui exames com outras alterações.

Essas diferenças **orientam a auditoria e o desenho experimental**. Não constituem motivo automático para descartar uma base ou manter a seleção documental indefinidamente aberta. Também não autorizam juntar rótulos de significados distintos sem justificativa.

## 5. Por que a fase 2 está concluída

- [x] Identificar e selecionar as três famílias inicialmente consideradas.
- [x] Incorporar duas famílias pediátricas à investigação.
- [x] Documentar origem, população, formatos, rótulos, metadados e caminhos de acesso.
- [x] Identificar possibilidades de uso e diferenças de referência.
- [x] Registrar a decisão final e encaminhar as verificações práticas para a fase 3.

O encerramento foi aprovado com esse escopo. Não implica que acesso, integridade, completude ou viabilidade de todos os experimentos já tenham sido comprovados.

## 6. Encaminhamento para a fase 3

| Frente da auditoria | O que verificar | Como poderá orientar as etapas seguintes |
|---|---|---|
| Acesso e versão | Pacote efetivamente obtido, termos, versão e arquivos disponíveis | Fixar a procedência dos dados analisados. |
| Formatos e conteúdo | Leitura dos arquivos, projeção, presença da região torácica/pulmonar e exclusões necessárias | Definir elegibilidade e identificar problemas de conteúdo. |
| Pacientes e estudos | Vínculo entre imagens, estudos e pacientes pseudonimizados | Viabilizar separação por paciente e evitar vazamento. |
| Duplicações | Repetições exatas, imagens semelhantes e sobreposição entre versões | Evitar contagens e avaliações indevidas. |
| População | Idade, precisão da idade, ausências e composição dos grupos | Verificar quais análises populacionais são sustentadas. |
| Referência | Rótulos disponíveis, incertezas, conflitos e procedência da anotação | Preparar a correspondência operacional entre bases. |
| Aquisição e qualidade | Metadados realmente preenchidos, unidades e informações de limitações técnicas | Identificar o que pode ser medido e interpretado. |
| Apresentação | Resolução, profundidade, conversões e processamento já aplicado | Separar características do arquivo de aspectos da aquisição. |

A auditoria deve preservar os testes finais: inspecionar sua integridade e estrutura não significa utilizá-los para escolher modelos, limiares ou transformações por desempenho.

Divergências já identificadas — como contagens entre versões do CheXpert Plus e informações etárias da PediCXR — permanecem registradas nas fichas e serão confrontadas com os arquivos. Não são resultados de auditoria já obtidos.

Decisões sobre imagem inteira, recorte ou segmentação pulmonar continuam abertas. A auditoria poderá fornecer evidências para justificá-las.

## 7. Sequência do cronograma

**Fase 2 concluída → fase 3 → retorno à fase 0 → fase 4 → modelagens e avaliações.**

| Concluído | Fase | Atividade | Entrega / critério de conclusão | Prazo |
|---|---|---|---|---|
| ☑ | **2 — Datasets** | Selecionar três famílias e incorporar duas bases pediátricas | Origem, população, rótulos, metadados, acesso e compatibilidade preliminar documentados; seleção aprovada. | — |
| ☐ | **3 — Auditoria preliminar** | Verificar a viabilidade dos dados | Examinar formatos, pacientes, duplicações, qualidade e possibilidade de medir as características escolhidas, preservando os testes finais. | — |
| ☐ | **0 — Fechamento** | Fechar hipótese e escopo | Definir desfecho principal, hipóteses testáveis e limites das conclusões com base no que os dados permitem. | — |
| ☐ | **4 — Protocolo** | Definir avaliação e análise estatística | Registrar splits por paciente, modelos, métricas, limiares, perturbações, comparações e estimativas de incerteza. | — |

Este trecho atualiza a passagem entre fases; não substitui as demais etapas do cronograma geral. Os papéis A/B/C serão atribuídos posteriormente, conforme a comparação escolhida.

## 8. Documentos de estudo preservados

| Arquivo | Função |
|---|---|
| `01_RSNA.md` | Estudo da RSNA. |
| `02_CheXpert_original.md` | Estudo do original e de seu datasheet. |
| `03_CheXpert_Plus.md` | Estudo da extensão com laudos e metadados. |
| `04_PadChest.md` | Estudo do PadChest. |
| `05_PediCXR_VinDr_PCXR.md` | Estudo da PediCXR. |
| `06_Kermany_pediatrico.md` | Estudo da coleção pediátrica de Guangzhou. |
| `07_matriz_compatibilidade.md` | Análise documental e alternativas discutidas, ainda sem força de protocolo experimental. |
| `08_literatura_e_leitura_critica.md` | Bibliografia inicial e perguntas para leitura dos artigos. |
| `00_planejamento_revisado.md` | Registro anterior da discussão do planejamento. |

**Os estudos anteriores continuam úteis e não foram descartados. Este README prevalece sobre eles quanto ao status de conclusão da fase 2, à decisão final de seleção e ao foco aprovado.** Condições e alternativas anteriormente discutidas não devem ser interpretadas como decisões experimentais definitivas. As fichas serão atualizadas quando novas evidências justificarem correções; seus conteúdos não foram alterados nesta entrega.

## 9. Referências e locais oficiais

### RSNA

1. [RSNA Pneumonia Detection Challenge — página oficial, descrição e downloads](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
2. Shih et al. (2019). [Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia](https://doi.org/10.1148/ryai.2019180041).

### CheXpert e CheXpert Plus

3. Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031). [Página oficial CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/).
4. Garbin et al. (2021). [Structured dataset documentation: a datasheet for CheXpert](https://arxiv.org/abs/2105.03020).
5. [Stanford AIMI — CheXpert Plus, descrição e acesso oficial](https://aimi.stanford.edu/datasets/chexpert-plus).
6. Chambon et al. (2024). [CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats](https://arxiv.org/abs/2405.19538). [Versão v1 consultada nos estudos](https://arxiv.org/html/2405.19538v1).

### PadChest

7. [BIMCV — PadChest, descrição e acesso oficial](https://bimcv.cipf.es/bimcv-projects/padchest/).
8. Bustos et al. (2020). [PadChest: A large chest x-ray image dataset with multi-label annotated reports](https://doi.org/10.1016/j.media.2020.101797). [Preprint de 2019 consultado](https://arxiv.org/abs/1901.07441).

### PediCXR / VinDr-PCXR

9. [PhysioNet — VinDr-PCXR v1.0.0, descrição e acesso oficial](https://physionet.org/content/vindr-pcxr/1.0.0/).
10. Pham et al. (2023). [PediCXR: An open, large-scale chest radiograph dataset for interpretation of common thoracic diseases in children](https://doi.org/10.1038/s41597-023-02102-5). [Texto dos autores](https://arxiv.org/abs/2203.10612).

### Kermany / Guangzhou

11. [Mendeley Data — Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification, versão 2](https://data.mendeley.com/datasets/rscbjbr9sj/2). Para o DIAMANTE, interessa o componente de tórax.
12. Kermany et al. (2018). [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://doi.org/10.1016/j.cell.2018.02.010).
13. [Self-Supervised Learning for Building Robust Pediatric Chest X-ray Classification Models (2024)](https://arxiv.org/html/2409.00231v1). Estudo de uso de PediCXR e Kermany; consultar diferenças de subconjuntos descritas na ficha bibliográfica.

### Generalização: referência metodológica

14. Zech et al. (2018). [Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002683). Referência sobre mudança de domínio; não representa uma avaliação das cinco famílias aqui selecionadas.

As referências documentam o levantamento desta fase. A bibliografia não constitui revisão sistemática, e leitura parcial ou indisponibilidade de texto integral está explicitada nas fichas correspondentes. Acesso prático aos datasets e conferência dos arquivos pertencem à etapa seguinte.

## 10. Histórico de decisão

| Data | Registro |
|---|---|
| 08/10/2026 | Aline aprovou o encerramento documental da fase 2, mantendo RSNA, CheXpert/Plus e PadChest e incluindo PediCXR e Kermany. Estudos individuais preservados; este README consolida o foco em pneumonia, as decisões aprovadas e o encaminhamento à auditoria. |

## Fontes

RSNA Pneumonia Detection Challenge (2018)

https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018?utm_source=chatgpt.com

a datasheet for CheXpert
https://arxiv.org/abs/2105.03020?utm_source=chatgpt.com
C
heXpert Plus + imagem com LAUDOS 2024 
https://aimi.stanford.edu/datasets/chexpert-plus?utm_source=chatgpt.com

PadChest: 
https://arxiv.org/abs/1901.07441