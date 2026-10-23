# DIAMANTE — Parecer da auditoria da amostra PadChest

Data: 16/10/2026. Papel pretendido da base: candidata à generalização externa 2.

## Decisão

**Escopo confirmado por Aline: auditoria preliminar da PadChest baseada na amostra disponível (24 imagens) e na documentação oficial. A amostra permite conhecer a estrutura, a cadeia de anotação, os metadados e a leitura dos arquivos. A PadChest permanece candidata à generalização externa 2, com restrições e sem aprovação definitiva do subconjunto experimental.**

O pequeno número de imagens com pneumonia limita o uso desta amostra como teste de desempenho; não impede sua utilização para a auditoria preliminar proposta.

As verificações executadas tiveram correspondência e leitura completas dos 24 arquivos. Não foram detectadas duplicações exatas de arquivos ou pixels. Esses resultados não validam os rótulos clinicamente, não excluem quase duplicações e não se estendem à base completa.

A etapa de inspeção desta amostra pode ser encerrada com o escopo abaixo e uma correção pontual na verificação de tempo de exposição. Obter o CSV completo ou baixar a coleção inteira não é condição para concluir esta auditoria preliminar. Se a PadChest for posteriormente selecionada para o experimento, a disponibilidade e a integridade do subconjunto efetivamente usado precisarão ser verificadas.

## Evidência e escopo

Fonte dos resultados: saídas do Kaggle fornecidas por Aline em `Texto colado.txt`. Execução registrada em 16/10/2026, 13:56:59 UTC. Nenhum resultado desta ficha foi obtido executando localmente as imagens reais.

- Pacote: `raddar/padchest-chest-xrays-sample`.
- CSV: `chest_x_ray_images_labels_sample.csv`.
- Ambiente informado: Python 3.13.15; pandas 2.3.3; NumPy 2.1.3; Pillow 12.3.0.
- Versão do Kaggle: ainda não preenchida na variável `VERSAO_KAGGLE`; registrar a versão exibida na plataforma.
- SHA256: calculado na execução, mas truncado na saída colada. O valor completo deve ser preservado em `resumo_auditoria.json`.
- Seleção/representatividade: método de seleção da amostra não verificado.

## 1. Proveniência local e tamanho

| Medida verificada | Resultado |
|---|---:|
| Linhas × colunas do CSV | 24 × 35 |
| Imagens PNG locais | 24 |
| Estudos identificados | 24 |
| Pacientes identificados | 24 |
| Tamanho local total | 166,99 MiB |
| IDs de imagem, estudo ou paciente ausentes | 0 |
| Linhas inteiramente idênticas | 0 |
| Correspondências únicas entre linha e imagem | 24/24 — 100% |
| Arquivos sem referência no CSV | 0 |

Não houve nomes ambíguos ou imagens listadas sem arquivo local. O vínculo foi testado pelo nome do arquivo de `ImageID`.

## 2. Rótulos, texto e anotação

Todas as 24 listas de rótulos foram interpretadas pelo código e todas as 24 linhas têm texto em `Report`. Isso comprova presença e legibilidade da estrutura; não comprova completude clínica do laudo nem correção de cada anotação.

| Indicador | Resultado |
|---|---:|
| Imagens com etiqueta exata `pneumonia` | 3/24 — 12,5% |
| Imagens sem essa etiqueta exata | 21/24 — 87,5% |
| Imagens com etiqueta `normal` | 9/24 — 37,5% |
| Método `Physician` | 6/24 — 25% |
| Método `RNN_model` | 18/24 — 75% |
| Pneumonia com método `Physician` | 1 |
| Pneumonia com método `RNN_model` | 2 |

Essas proporções descrevem a amostra; não estimam prevalência da doença na população ou na base completa. O pacote é multilabel. A presença de `normal` precisa ser confrontada com coocorrências e regras do protocolo antes de definir negativos. As 21 imagens sem etiqueta exata `pneumonia` não foram classificadas como negativas confirmadas.

A documentação da PadChest descreve rótulos extraídos dos laudos, com anotação humana e automática de texto. O campo `Report` é descrito pelos autores como trecho processado. `RNN_model` não deve ser interpretado aqui como previsão de pneumonia produzida pelo nosso classificador de imagens. `Physician` também não equivale automaticamente a diagnóstico confirmado por exames clínicos independentes. [1, 2]

## 3. Projeções e relevância para o teste externo

| Projection | Total | Com etiqueta exata pneumonia |
|---|---:|---:|
| PA | 14 | 1 |
| AP | 1 | 0 |
| AP_horizontal | 1 | 0 |
| L | 7 | 1 |
| COSTAL | 1 | 1 |
| Total | 24 | 3 |

Com um critério de projeção limitado a PA/AP, restam 15 imagens e apenas uma com etiqueta exata de pneumonia. Se o protocolo também aceitar AP_horizontal como frontal, restam 16 e continua apenas uma. Esses números são anteriores a qualquer filtro por idade, origem de rótulo ou qualidade.

Portanto, no subconjunto frontal, há **no máximo uma imagem positiva candidata**; ainda não foi cruzada sua idade e origem da anotação. Um teste com uma única positiva só permitiria sensibilidade empírica de 0% ou 100%, sem precisão útil para a avaliação pretendida. Se o caso for inelegível, o subconjunto ficará sem positivos.

`MethodProjection` registra 22 imagens como `Manual review of DICOM fields` e duas como `resnet-50.t7`. A origem da projeção é diferente da origem do rótulo de doença: são dois processos distintos, que precisam de registros separados. As duas projeções inferidas automaticamente não foram validadas visualmente nesta auditoria.

## 4. População e cobertura dos metadados

- Sexo: F = 11 (45,8%); M = 13 (54,2%).
- Idade aproximada: mínimo 10, máximo 83, média 53,875, mediana 58 anos. Foi calculada pela diferença entre anos de exame e nascimento, sem datas completas; requer conferência do formato para decisões de elegibilidade.
- Existe ao menos um caso com idade aproximada pediátrica. A amostra não pode ser tratada integralmente como adulta.
- `Pediatric` está vazio em 24/24. Isso significa ausência de informação nessa coluna, não ausência de crianças.
- Fabricantes: ImagingDynamicsCompanyLtd = 11; PhilipsMedicalSystems = 13. Há identificação em 24/24, mas não foi estabelecida uma relação causal entre equipamento e rótulo.

| Campo | Preenchidos | Ausentes |
|---|---:|---:|
| Projection / MethodProjection | 24/24 | 0% |
| ViewPosition_DICOM | 15/24 | 37,5% |
| PatientBirth / StudyDate_DICOM / PatientSex_DICOM | 24/24 | 0% |
| Manufacturer_DICOM / Modality_DICOM | 24/24 | 0% |
| PhotometricInterpretation_DICOM / BitsStored_DICOM | 24/24 | 0% |
| Rows_DICOM / Columns_DICOM | 24/24 | 0% |
| XRayTubeCurrent_DICOM | 15/24 | 37,5% |
| Exposure_DICOM | 19/24 | 20,8% |
| ExposureInuAs_DICOM | 15/24 | 37,5% |
| RelativeXRayExposure_DICOM | 21/24 | 12,5% |
| Pediatric | 0/24 | 100% |
| ExposureTime | Coluna presente | Cobertura ainda não calculada na saída recebida |

**Correção do notebook:** a versão executada procurou `ExposureTime_DICOM`, mas a lista de colunas recebida contém `ExposureTime`. A ausência do primeiro nome não comprova ausência de tempo de exposição. A taxa de 100% registrada para `ExposureTime_DICOM` não deve ser apresentada como resultado da variável física. O notebook foi corrigido para reconhecer o nome presente; ainda é preciso calcular a cobertura e verificar unidades/valores. Valores preenchidos também podem ser inválidos ou representar códigos de ausência.

## 5. Integridade e pixels

- As 24 imagens foram lidas sem falha na execução apresentada.
- SHA256 dos arquivos: zero grupos idênticos.
- Hash de dimensões, dtype e pixels: zero grupos idênticos.
- Os cinco exemplos exibidos em `pixels.head()` são PNG, modo `I;16`, dtype `uint16`, com dimensões variadas. O texto recebido não exibe a tabela completa; não atribuímos automaticamente essas características às 24 imagens.
- O código calculou mínimo, máximo, média, desvio e percentis para os arrays bidimensionais; os resultados completos foram salvos em `08_integridade_pixels.csv`.
- Não houve inspeção clínica das imagens nem validação de exposição física, orientação, fidelidade à conversão DICOM ou correspondência completa das dimensões do CSV com os PNGs.

Intensidade de pixel em PNG não equivale diretamente à exposição física. Campos DICOM no CSV descrevem a origem; não são cabeçalhos DICOM presentes no próprio PNG.

## 6. Dependência e independência

Os IDs fornecidos identificam uma imagem e um estudo por paciente na amostra: 24/24 únicos, nenhum paciente com mais de um estudo e nenhum StudyID associado a vários PatientID. Isso não demonstra que a PadChest completa não possua exames repetidos.

Sobreposição com RSNA, CheXpert, pré-treinamento de modelos externos e quase duplicações não foram verificadas.

## Implicações para o DIAMANTE

| Uso | Parecer |
|---|---|
| Verificar leitura, esquema e execução da auditoria | Viável no pacote recebido; 24 imagens lidas e correspondência completa. |
| Entender a cadeia de anotação e os campos disponíveis | Útil; há texto, método de rótulo e método de projeção identificáveis. |
| Treinar ou selecionar o modelo com esta amostra | Sem justificativa no desenho atual; a finalidade candidata da PadChest é avaliação externa. |
| Usar esta amostra como teste final de pneumonia | Inadequado para o objetivo: dimensão pequena, seleção não demonstrada e no máximo uma positiva candidata após filtro frontal. |
| Excluir a PadChest completa por esses resultados | Não sustentado. A restrição da amostra não demonstra inadequação da base completa. |
| Aprovar a PadChest completa para teste externo | Evidência insuficiente. Manter como candidata condicionada à auditoria do subconjunto real. |

## Encerramento dentro do escopo escolhido

1. Corrigir o registro de tempo de exposição e preencher a versão do pacote; preservar as saídas já produzidas.
2. Consolidar a ficha científica com as evidências da amostra e da documentação, identificando claramente a origem de cada resposta. Registrar como não verificado tudo o que depender de dados indisponíveis.
3. Encerrar como **auditoria preliminar por amostra**, com PadChest mantida como candidata com restrições. A integridade observada refere-se às 24 imagens e as frequências calculadas descrevem apenas a amostra.

Não é necessário baixar a base completa ou obter seu CSV para esse encerramento. A amostra não precisa funcionar como teste final para ser útil à auditoria preliminar.

## Verificações futuras, somente se a PadChest for escolhida para uso experimental

Na definição do subconjunto efetivamente utilizado, verificar número de pacientes elegíveis, projeções, referência positiva e negativa, origem da anotação, disponibilidade e leitura das imagens, dependência e independência em relação ao desenvolvimento. O acesso ao CSV completo poderá ajudar nessa etapa, mas não é uma tarefa obrigatória agora.

Se acesso ou tempo inviabilizarem o subconjunto experimental, registrar o adiamento por viabilidade operacional. Isso não constitui rejeição científica de toda a base.

## Fontes documentais

1. BIMCV: https://bimcv.cipf.es/bimcv-projects/padchest/
2. Autores — dicionário e métodos: https://github.com/auriml/Rx-thorax-automatic-captioning
3. Artigo: https://doi.org/10.1016/j.media.2020.101797
4. Pacote auditado: https://www.kaggle.com/datasets/raddar/padchest-chest-xrays-sample

Os números deste parecer vêm das saídas fornecidas, não das contagens globais dessas páginas.
