# Relatório final de auditoria — RSNA Pneumonia Detection Challenge

**Projeto:** IC Pneumonia — auditoria de viabilidade da base RSNA  
**Notebook de origem:** 01-auditoria-rsna-ic.ipynb  
**Data da auditoria:** 13/10/2026  
**Status:** auditoria de dados concluída; **aprovada com restrições** como candidata à base A (desenvolvimento).

## 1. Objetivo e escopo

Verificar a estrutura, integridade, rótulos, caixas delimitadoras, metadados e características descritivas de apresentação do pacote RSNA Stage 2 disponibilizado no Kaggle, antes de treinar modelos.

A auditoria de dados está encerrada. Ela não treinou modelos, não modificou imagens e não avaliou desempenho preditivo. A independência entre pacientes não foi comprovada por uma contagem direta no Stage 2; essa restrição está documentada e deve ser resolvida no protocolo de divisão antes de qualquer treinamento.

## 2. Proveniência e interpretação de “dataset completo”

O Kaggle disponibilizou 26.684 DICOMs de treino e 3.000 DICOMs de teste, totalizando 29.684 imagens. O pacote é compatível com o conjunto completo distribuído para o desafio Stage 2: todos os IDs de treino correspondem aos rótulos e todos os IDs de teste correspondem à lista de referência do teste.

Isso não significa que a base contenha todas as imagens do NIH CXR8. A RSNA descreve o conjunto do desafio como aproximadamente 30 mil radiografias selecionadas de uma coleção NIH de aproximadamente 112 mil imagens. Portanto, o conjunto do desafio é um recorte da coleção NIH mais ampla.

### Inventário observado no Kaggle

| Componente | Quantidade observada |
|---|---:|
| DICOMs de treino | 26.684 |
| DICOMs de teste | 3.000 |
| Total de DICOMs | 29.684 |
| Arquivos CSV | 3 |
| Arquivo TXT | 1 |
| Total de arquivos reportado no diretório | 29.688 |

## 3. Integridade e correspondência de arquivos

Foram verificadas as correspondências entre arquivos DICOM e identificadores de referência:

| Verificação | Resultado |
|---|---:|
| Imagens de treino sem ID nos rótulos | 0 |
| IDs de rótulo sem imagem de treino | 0 |
| Imagens de teste sem ID na amostra de referência | 0 |
| IDs da amostra de referência sem imagem de teste | 0 |

Todos os 29.684 DICOMs puderam ser lidos sem falhas. As imagens de treino e teste têm dimensões de 1024 × 1024 pixels.

**Interpretação:** não foram encontrados arquivos ausentes nas correspondências verificadas para o pacote Stage 2. Essa verificação não substitui a investigação de duplicatas entre imagens ou o agrupamento por paciente de origem.

## 4. Rótulos e classes

O arquivo de rótulos tem 30.227 linhas para 26.684 IDs únicos. Há 3.543 ocorrências adicionais de IDs no arquivo, compatíveis com múltiplas caixas delimitadoras associadas a uma imagem positiva. Nenhum ID apresentou mais de um valor de Target.

O arquivo de classes detalhadas também contém 30.227 linhas e 26.684 IDs únicos. Após consolidar por ID, não foi encontrado nenhum caso com classes detalhadas conflitantes. A junção por ID foi completa.

### Distribuição por imagem/ID único no treino

| Classe detalhada | Target | Imagens/IDs |
|---|---:|---:|
| Lung Opacity | 1 | 6.012 |
| No Lung Opacity / Not Normal | 0 | 11.821 |
| Normal | 0 | 8.851 |
| **Total** |  | **26.684** |

**Implicação metodológica:** Target = 0 reúne duas classes distintas: Normal e No Lung Opacity / Not Normal. Logo, a tarefa binária do desafio não deve ser descrita automaticamente como “pneumonia versus normal”. A classe Lung Opacity é uma anotação de opacidade pulmonar possivelmente compatível com pneumonia, e não equivale, por si só, a confirmação clínica de pneumonia.

O conjunto de teste não tem rótulos públicos para essa análise; seus 3.000 casos foram usados apenas nas verificações de estrutura e IDs.

## 5. Caixas delimitadoras

As quatro coordenadas — x, y, width e height — estão ausentes nas 20.672 linhas negativas, como esperado, e preenchidas em todas as linhas positivas. Foram encontradas 9.555 linhas de caixas para 6.012 IDs positivos.

| Verificação das caixas positivas | Resultado |
|---|---:|
| Caixas com posição x ou y negativa | 0 |
| Caixas com largura ou altura não positiva | 0 |
| DICOMs positivos lidos para obter dimensões | 6.012 |
| Falhas de leitura desses DICOMs | 0 |
| Caixas sem dimensão DICOM associada | 0 |
| Caixas que ultrapassam os limites da imagem | 0 |

Cada caso positivo tem entre 1 e 4 caixas delimitadoras; média de 1,59 e mediana de 2 caixas por caso.

**Conclusão:** as caixas têm consistência geométrica básica em relação às dimensões DICOM. Esta checagem não avalia a precisão clínica das caixas.

## 6. Metadados DICOM

Foram lidos os cabeçalhos de todos os 29.684 DICOMs, sem falhas.

| Campo | Disponibilidade / distribuição |
|---|---|
| Rows / Columns | 100% preenchidos; todas as imagens 1024 × 1024 |
| PatientAge | 100% preenchido |
| PatientSex | M: 16.880; F: 12.804 |
| ViewPosition | AP: 13.555; PA: 16.129 |
| PhotometricInterpretation | MONOCHROME2 em 100% |
| BitsAllocated / BitsStored | 8 / 8 em 100% |
| PixelRepresentation | 0 em 100% |
| PixelSpacing | 100% preenchido |
| KVP, Exposure, ExposureTime, XRayTubeCurrent, Manufacturer | ausentes em 100% |

A ausência de parâmetros de exposição e fabricante impede uma comparação direta de técnicas e equipamentos com base nesses campos. As imagens do desafio são descritas pela RSNA como imagens PNG convertidas para DICOM, com idade, sexo e projeção adicionados às tags. Portanto, as tags disponíveis não devem ser tratadas automaticamente como um registro completo da aquisição original do equipamento.

### Idades

A idade foi parseada dos cabeçalhos para fins descritivos. No conjunto completo, foram encontrados seis valores acima de 120 anos: cinco no treino (148, 150, 151, 153 e 155) e um no teste (412). O valor 412 aparece no teste, sem rótulo público. Esses valores devem ser mantidos como dados brutos e sinalizados como anomalias; não devem ser interpretados como idades plausíveis nem corrigidos sem investigar a origem.

No treino, 1.276 imagens/IDs estão na faixa de 0–17 anos, aproximadamente 4,8% do conjunto rotulado. A RSNA não é exclusivamente adulta.

## 7. Composição por projeção, sexo e idade

Foi observada uma forte associação entre projeção e classe no treino:

| Projeção | Lung Opacity | No Lung Opacity / Not Normal | Normal |
|---|---:|---:|---:|
| AP | 4.664 (38,3%) | 5.872 (48,2%) | 1.637 (13,4%) |
| PA | 1.348 (9,3%) | 5.949 (41,0%) | 7.214 (49,7%) |

As proporções por sexo foram mais próximas:

| Sexo | Lung Opacity | No Lung Opacity / Not Normal | Normal |
|---|---:|---:|---:|
| F | 21,7% | 44,4% | 33,9% |
| M | 23,1% | 44,2% | 32,6% |

As classes também variam entre grupos etários. Entre os casos com idade agrupável no treino, a proporção de Lung Opacity foi 30,2% em 0–17 anos, 24,2% em 18–39, 21,7% em 40–59 e 20,6% em 60–120.

**Implicação metodológica:** a diferença por projeção é um possível fator de confusão. Um modelo futuro pode aprender associações com AP/PA correlacionadas aos rótulos. A auditoria identifica essa associação, mas não demonstra causalidade.

## 8. Auditoria exploratória de intensidade dos pixels

Foi selecionada, com semente fixa 42, uma amostra estratificada de 100 imagens para cada combinação de classe detalhada e projeção, totalizando 600 imagens. Foram calculadas estatísticas dos valores de pixels da imagem inteira, sem alterar os dados.

### Mediana da média dos pixels por imagem

| Classe | AP | PA |
|---|---:|---:|
| Lung Opacity | 116,10 | 119,36 |
| No Lung Opacity / Not Normal | 122,13 | 122,01 |
| Normal | 125,51 | 128,16 |

As medianas foram um pouco menores para Lung Opacity do que para Normal nas duas projeções, mas as distribuições dos boxplots se sobrepõem amplamente. A média global dos pixels não separa claramente as classes.

A medida inclui margens e fundo da imagem; por isso, é um descritor exploratório da apresentação dos pixels armazenados, não uma medida clínica de qualidade ou de exposição física. A amostra balanceada serve para comparar estratos, não para estimar a prevalência de classes.

## 9. Identificação de pacientes e prevenção de leakage

Os `patientId` do Kaggle identificam imagens do desafio, não garantem um paciente por ID. A publicação original relata 30.000 exames e 12.274 pacientes únicos, com até 75 radiografias para um paciente. Isso comprova que há exames repetidos por paciente no conjunto de origem e torna inadequada uma divisão aleatória por imagem. Esses números são da coorte publicada de 30.000 exames; não substituem a contagem exata do pacote Stage 2 do Kaggle (29.684 imagens).

A RSNA disponibiliza oficialmente um arquivo de mapeamento entre os identificadores do desafio e as imagens NIH originais. A existência do arquivo foi confirmada na página oficial da RSNA. O mapeamento não foi aplicado nesta execução: o JSON oficial e os metadados NIH não estavam entre os arquivos anexados ao notebook. Portanto, não foram calculados a cobertura do mapeamento para os 29.684 IDs do Stage 2, o número exato de pacientes representados, nem a sobreposição de pacientes entre as partições públicas de treino e teste.

**Regra para a etapa de modelagem:** obter o mapeamento oficial e agrupar as imagens pelo identificador de paciente NIH de origem; separar treino e validação por paciente, mantendo todos os exames de cada paciente no mesmo grupo. O teste público do desafio não deve ser tratado como teste interno rotulado, pois os rótulos não estão disponíveis para esta auditoria. A comparação de duplicatas entre bases também permanece fora do escopo executado.

## 10. Parecer final

**Decisão: auditoria de dados concluída; RSNA aprovada com restrições como candidata à base A — desenvolvimento.**

A RSNA tem arquivos íntegros e rótulos/caixas utilizáveis para desenvolver uma tarefa definida a partir das anotações do desafio. Entretanto, antes de fixar o protocolo de treino e validação, precisam ser tratadas as seguintes limitações:

1. Definir operacionalmente o desfecho, sem chamar Target = 0 de “Normal” e sem tratar opacidade anotada como diagnóstico clínico confirmado.
2. Antes de treinar, aplicar o mapeamento oficial RSNA–NIH e executar a divisão de treino/validação por paciente; a contagem exata de pacientes e a sobreposição entre partições do Stage 2 permanecem não calculadas nesta auditoria.
3. Verificar duplicatas de imagens, inclusive entre divisões e em relação a outras bases, quando a comparação externa avançar.
4. Documentar que fabricante e parâmetros de exposição não estão disponíveis nos DICOMs do pacote.
5. Investigar os seis valores de idade acima de 120, preservando os valores originais.
6. Expandir a análise de pixels além da amostra estratificada antes de formular conclusões sobre diferenças de apresentação.
7. Definir uma inspeção sistemática de artefatos visuais, se fizer parte do protocolo.

Esta auditoria não autoriza ainda inferências sobre desempenho, erro, calibração ou generalização de modelos. Nenhum modelo foi treinado nesta fase.

## 11. Fontes

- RSNA. **RSNA Pneumonia Detection Challenge (2018)** — descrição, arquivos, mapeamento oficial para NIH e termos de uso.  
  https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
- RSNA. **Arquivo de mapeamento RSNA–NIH** (JSON).  
  https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/pneumonia-challenge-dataset-mappings_2018.json
- Shih G. et al. **Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.** *Radiology: Artificial Intelligence*, 2019.  
  https://doi.org/10.1148/ryai.2019180041
- Kaggle. **RSNA Pneumonia Detection Challenge — Data** — arquivos Stage 2 e documentação do desafio.  
  https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data
- RSNA ATLAS AI Data Card — informações de proveniência e composição do conjunto.  
  https://atlas.rsna.org/cards/74777c92-9b0b-4f06-a788-71a8a12c4e93
