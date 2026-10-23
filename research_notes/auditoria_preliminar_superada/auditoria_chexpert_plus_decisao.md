# Auditoria preliminar do CheXpert Plus

**Projeto:** DIAMANTE — auditoria de robustez e generalização de modelos para pneumonia em radiografias de tórax  
**Data da auditoria:** 14 de outubro de 2026  
**Fonte:** dataset CheXpert Plus, Stanford AIMI, versão 1.0 na Redivis  
**DOI:** [10.71718/6nvz-pm34](https://doi.org/10.71718/6nvz-pm34)  
**Página do dataset:** [CheXpert Plus na Redivis](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next)

## 1. Objetivo e escopo

Esta auditoria preliminar avaliou se o CheXpert Plus pode contribuir para o DIAMANTE, especialmente como candidato a avaliação externa de um modelo desenvolvido na RSNA. Foram examinados estrutura, correspondência entre imagens e índices de arquivos, cobertura DICOM direta e compactada, rótulos de pneumonia e consistência entre seções dos relatórios.

O trabalho ainda não avaliou a integridade binária nem a decodificação dos DICOMs que aparecem somente dentro dos ZIPs. Também não realizou uma auditoria completa dos pixels das imagens, de sua qualidade radiográfica ou de todos os metadados DICOM. Portanto, este documento encerra a **triagem de viabilidade e cobertura por caminhos**, não a validação técnica integral de cada arquivo.

## 2. Descrição da base

Conforme a descrição do dataset, o CheXpert Plus reúne **223.462 pares de imagem e relatório**, provenientes de **187.711 estudos** e **64.725 pacientes**. A base oferece imagens PNG, imagens DICOM, relatórios radiológicos e rótulos de patologias.

Na Redivis, a tabela principal `df_chexpert_plus_240401` possui 223.462 linhas e 27 colunas. As três tabelas JSON examinadas — `findings_fixed.json`, `impression_fixed.json` e `report_fixed.json` — têm 223.462 registros cada, com caminhos únicos e sem caminhos ausentes.

## 3. Resultados da auditoria

### 3.1 Cobertura de imagens e correspondência de caminhos

| Verificação | Resultado |
|---|---:|
| PNGs em `PNG_train` | 223.228 |
| PNGs em `PNG_valid` | 234 |
| Caminhos PNG únicos | 223.462 |
| Caminhos PNG sem rótulo correspondente | 0 |
| Rótulos sem PNG correspondente | 0 |

Os 223.462 caminhos PNG correspondem exatamente aos caminhos das tabelas de rótulos examinadas.

### 3.2 Cobertura DICOM direta e nos ZIPs

Os índices diretos contêm **210.969 DICOMs**: 210.740 em `DICOM_train` e 229 em `DICOM_valid`. Inicialmente, 12.493 caminhos rotulados não foram encontrados nesses dois índices.

Os 18 arquivos de `DICOM_compressed` foram examinados por seus índices internos, usando leituras parciais de bytes. Não houve erros na leitura dos índices.

| Comparação de caminhos | Resultado |
|---|---:|
| Caminhos DICOM únicos listados nos ZIPs | 223.462 |
| Caminhos dos ZIPs também presentes no índice DICOM direto | 210.969 |
| Caminhos listados nos ZIPs e ausentes do índice direto | 12.493 |
| Caminhos PNG sem correspondência nos ZIPs | 0 |
| Caminhos extras nos ZIPs em relação às PNGs | 0 |
| Conjunto de ausências do índice direto igual ao conjunto ZIP-only | Sim |

Assim, a lacuna de 12.493 caminhos é explicada pela organização dos arquivos: eles não aparecem em `DICOM_train`/`DICOM_valid`, mas estão listados nos índices de `DICOM_compressed`. A verificação confirma **presença dos caminhos nos índices internos dos ZIPs**; não confirma ainda que todos os conteúdos foram extraídos ou decodificados com sucesso.

### 3.3 Estudos afetados pela ausência no índice direto

| Indicador | Resultado |
|---|---:|
| Imagens sem DICOM no índice direto | 12.493 (5,59% de 223.462) |
| Pacientes afetados | 1.779 |
| Estudos afetados | 10.790 (5,75% de 187.711) |
| Estudos com DICOM direto para todas as imagens | 176.921 |
| Estudos com DICOM direto para algumas imagens | 0 |
| Estudos sem DICOM direto para nenhuma imagem | 10.790 |

As ausências são agrupadas por estudo: não foram observados estudos parcialmente cobertos pelo índice DICOM direto.

| Vista | Sem DICOM direto / total | Taxa ausente |
|---|---:|---:|
| `view1_frontal` | 10.787 / 187.670 | 5,75% |
| `view1_lateral` | 1 / 16 | 6,25% |
| `view2_frontal` | 189 / 3.360 | 5,62% |
| `view2_lateral` | 1.435 / 30.637 | 4,68% |
| `view3_frontal` | 5 / 41 | 12,20% |
| `view3_lateral` | 76 / 1.738 | 4,37% |

O percentual de `view3_frontal` deve ser interpretado com cautela, pois o denominador é pequeno (41 imagens). A maior contagem absoluta em `view1_frontal` acompanha sua maior frequência na base; as taxas por vista são, em geral, próximas.

### 3.4 Rótulos de pneumonia

Os estados observados nos arquivos de rótulos são `1` (positivo), `0` (negativo), `-1` (incerto) e `null` (sem rótulo). `null` não foi tratado como negativo.

| Seção de origem | Sem rótulo (`null`) | Incerto (`-1`) | Negativo (`0`) | Positivo (`1`) | Positivos + negativos definidos |
|---|---:|---:|---:|---:|---:|
| `findings` | 218.464 | 3.702 | 381 | 915 | 1.296 |
| `impression` | 195.926 | 19.367 | 3.322 | 4.847 | 8.169 |
| `report` | 189.734 | 23.045 | 4.720 | 5.963 | 10.683 |

Em `impression`, há 8.169 rótulos definidos de pneumonia (3,66% dos 223.462 caminhos), além de 19.367 incertos e 195.926 sem rótulo. A escolha da seção e o tratamento dos incertos e ausentes mudam substancialmente o conjunto avaliável.

Na tabela principal, `section_findings` contém texto em 59.441 linhas (aproximadamente 26,6%) e `section_impression` em 223.296 (aproximadamente 99,9%).

### 3.5 Divergência entre fontes de rótulo

Foram comparados os estados não nulos de Pneumonia entre os três arquivos.

| Par de fontes | Com rótulo nas duas | Iguais | Divergentes |
|---|---:|---:|---:|
| `findings` × `impression` | 4.096 | 3.653 | 443 |
| `findings` × `report` | 746 | 396 | 350 |
| `impression` × `report` | 4.162 | 2.210 | 1.952 |

Uma revisão manual pequena dos casos divergentes encontrou exemplos em que o contexto clínico do relatório e a conclusão da impressão diferem. Essa inspeção ilustra a divergência, mas não determina sua causa nem valida qual rótulo está correto. Nesta auditoria, os rótulos associados às seções do relatório não foram validados contra uma referência clínica independente.

## 4. Interpretação para o DIAMANTE

O CheXpert Plus tem valor para o projeto porque oferece outro domínio institucional, imagens associadas a relatórios e estrutura de estudos/pacientes. A correspondência de caminhos está completa para PNGs e para os membros catalogados nos ZIPs. Isso permite considerar a base para avaliação externa de um modelo desenvolvido na RSNA, sujeita à definição prévia de uma política de rótulos.

O principal limitador para avaliação quantitativa é a referência de pneumonia: há grande proporção de valores ausentes ou incertos e divergência relevante entre `findings`, `impression` e `report`. Usar todos os registros como se tivessem rótulos binários confiáveis produziria uma comparação inadequada.

## 5. Decisão

### Decisão da triagem

**Manter o CheXpert Plus como candidato condicional a avaliação externa (papel B), sem usá-lo inicialmente para treinamento.** A auditoria não justifica desclassificá-lo: há cobertura de caminhos integral, e a base oferece um domínio externo e relatórios que podem apoiar a pesquisa.

### Condições para uso no experimento principal

1. Definir antes da avaliação qual fonte de rótulo será usada. `impression` pode ser avaliada como candidata por representar a conclusão do relatório, mas ainda não está aprovada como padrão-ouro.
2. Definir previamente como tratar `null` e `-1`. Eles não devem ser convertidos automaticamente em negativos.
3. Manter o modelo e os parâmetros congelados antes do teste externo; CheXpert Plus não deve orientar ajuste de hiperparâmetros ou limiar após a inspeção dos resultados.
4. Agrupar análises e divisões por paciente quando houver qualquer particionamento interno, evitando separar imagens do mesmo paciente entre conjuntos.
5. Se forem necessários atributos nativos do DICOM, validar a extração e leitura de uma amostra de arquivos que aparecem somente nos ZIPs. Para análise com PNG, a cobertura de caminhos foi confirmada, mas a auditoria de pixels ainda precisa ser feita.

### Sobre uma nota percentual de contribuição

Não foi atribuída uma porcentagem de “contribuição ao projeto”. A auditoria não define uma escala validada para essa medida; um corte de 80% exigiria critérios e pesos estabelecidos antes de pontuar todos os datasets. A decisão nesta etapa é categórica e condicionada: **manter como candidato externo, sujeito à compatibilidade dos rótulos e ao protocolo final**. Uma ficha comparativa comum deve ser aplicada às demais bases antes da seleção definitiva.

## 6. O que esta auditoria não concluiu

- Não comprovou CRC, extração ou decodificação de todos os 12.493 DICOMs que aparecem somente nos ZIPs.
- Não verificou campos e completude dos 47 atributos DICOM descritos para a base.
- Não avaliou sistematicamente resolução, orientação, intensidade, contraste, artefatos ou qualidade radiográfica dos pixels.
- Não estabeleceu equivalência clínica dos rótulos com RSNA ou PadChest.
- Não selecionou definitivamente a regra de rótulo para o experimento externo.

## 7. Próximo passo

Encerrar a triagem preliminar de caminhos do CheXpert Plus e prosseguir para a auditoria do PadChest usando uma ficha comparável. Depois, comparar RSNA, CheXpert Plus e PadChest para decidir os papéis finais e, se CheXpert Plus continuar selecionado, retornar apenas aos testes necessários para validar rótulos e acesso aos DICOMs.

## 8. Fontes

- [Página oficial do CheXpert Plus na Redivis](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next)
- [DOI do CheXpert Plus](https://doi.org/10.71718/6nvz-pm34)
- Resultados computacionais desta auditoria, executados no notebook Redivis em 14/10/2026.
