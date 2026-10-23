# DIAMANTE — RSNA Pneumonia Detection Challenge

Data: 08/10/2026. Selecionada para auditoria. Evidência documental; arquivos não inspecionados.

## Origem, população e formato

| Campo | Informação documentada |
|---|---|
| Origem | Subconjunto do NIH Clinical Center, EUA, reanotado para o desafio de 2018. [R1, R2] |
| Dimensão | Coleção de 30.000 exames; 12.274 pacientes únicos. O artigo relata 106 exclusões da anotação. [R2] |
| Idades | Há adultos e menores: 28.255 e 1.639 exames, respectivamente, no conjunto descrito após exclusões. Não são contagens de pacientes únicos. [R2] |
| Projeções | Frontais AP e PA. [R3] |
| Formato | PNG do NIH convertido para DICOM; idade, sexo e projeção foram acrescentados aos cabeçalhos. [R3] |

## Significado da referência

O alvo é **opacidade suspeita de pneumonia no contexto clínico apropriado**, anotada visualmente por especialistas. Não é confirmação clínica ou microbiológica. A instrução de considerar tosse e febre era uma premissa da leitura, não sintomas registrados de cada paciente. [R2]

O artigo distingue normal, alteração sem opacidade suspeita de pneumonia e opacidade suspeita. A categoria “No Opacity/Not Normal” pode conter opacidade de outra natureza. Portanto, não significa ausência de qualquer opacidade. O conjunto tem versões com e sem adjudicação. [R2]

## Interpretação proposta para o DIAMANTE

| Possível tarefa | Positivos | Negativos | Limitação |
|---|---|---|---|
| Detectar opacidade suspeita | Referência positiva do desafio | Normal + alterações sem opacidade suspeita | Preservar os dois subgrupos negativos na avaliação. |
| Comparar suspeita versus normal | Referência positiva | Apenas normal | Tarefa mais restrita; não representa o espectro completo de exames anormais. |

Essas definições são propostas metodológicas. A convenção exata do pacote escolhido e das caixas deve ser conferida antes da implementação. Ausência de caixa em um arquivo incompleto não é automaticamente um negativo.

## Identidade, laudos e aquisição

A RSNA oferece um mapeamento entre identificadores de imagens do desafio e arquivos originais do NIH. Esse é o caminho documentado para recuperar o agrupamento pseudonimizado por paciente, usando os metadados da origem. [R1] O nome `patientId` do desafio não será aceito como prova suficiente de identidade longitudinal.

O pacote consultado não oferece os laudos clínicos completos como componente descrito. A procedência NIH não torna a anotação final RSNA uma simples extração de laudos. [R1, R2]

O DICOM distribuído não recupera informação de aquisição perdida na conversão anterior. [R3] Para o projeto, isso significa que aparência do pixel poderá ser investigada, mas não será tratada como medida direta de exposição ou configuração do detector.

## Acesso e decisão

Entrada oficial: [RSNA — página do desafio](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018). A página fornece imagens, anotações, mapeamento e termos de atribuição. [R1]

**Manter** pela referência visual e localização. A participação pediátrica será determinada pelos metadados; não substituirá automaticamente uma base infantil dedicada.

## Pendências para a fase 3

- Identificar versão, etapa do desafio e quantidade efetivamente acessível e rotulada.
- Executar e verificar o vínculo com o identificador NIH de paciente; registrar correspondências ausentes.
- Conferir intervalo etário, unidade, repetições e extremos implausíveis.
- Separar referências adjudicadas das demais, se os arquivos permitirem.
- Verificar sobreposição com modelos pré-treinados em NIH/ChestX-ray14.

## Fontes

- **R1.** [RSNA — recurso oficial e downloads](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).
- **R2.** Shih et al. (2019). [Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia](https://doi.org/10.1148/ryai.2019180041). Artigo de construção; consultar “Assumptions” e “Resulting Datasets”.
- **R3.** [Descrição oficial do dataset, PDF](https://www.rsna.org/artificial-intelligence/-/media/Files/RSNA/Education/AI%20resources%20and%20training/AI%20image%20challenge/RSNA-2018-Pneumonia-Detection-Challenge-Dataset-Description.ashx). Disponível também em “Dataset description” da página R1.

Literatura de aplicação: [Pneumonia Detection in Chest Radiographs, DeepRadiology Team, 2018](https://arxiv.org/abs/1811.08939). Demonstra uso no desafio; não estabelece validade pediátrica do nosso futuro modelo.
