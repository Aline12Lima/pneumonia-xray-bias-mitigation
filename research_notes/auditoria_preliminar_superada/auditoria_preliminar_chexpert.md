# DIAMANTE — Ficha científica padronizada: CheXpert

**Revisão:** 16/10/2026  
**Execução local documentada:** 14/10/2026  
**Padrão adotado:** `Ficha_cientifica_datasets_DIAMANTE.md`, versão 16/10/2026.  
**Decisão atual:** manter como candidata à base de desenvolvimento.  
**Status da auditoria:** em andamento; exploração inicial dos CSVs realizada, verificações complementares e auditoria das imagens pendentes.

## 1. Objetivo e alcance desta revisão

Responder: **o pacote disponível é íntegro e adequado ao papel experimental proposto, e quais limites precisam entrar no protocolo?**

Esta revisão aplica as 24 perguntas da ficha comum aos resultados existentes do CheXpert e à documentação consultada. As saídas foram produzidas pela pesquisadora no notebook local e compartilhadas durante a auditoria. Os CSVs completos e o notebook executado não foram anexados para reexecução nesta revisão. Os números apresentados são aqueles resultados, com conferência aritmética dos totais.

**Correção do parecer anterior:** a organização deste relatório está concluída, mas a auditoria padronizada do pacote CheXpert permanece em andamento. A análise inicial dos CSVs não comprova a integridade das imagens nem equivale ao conjunto de verificações já realizado na RSNA. Não houve treinamento, avaliação preditiva ou nova execução das análises sobre os dados brutos nesta revisão.

## 2. Identificação do pacote e rastreabilidade

| Campo | Registro |
|---|---|
| Dataset | CheXpert, publicado originalmente em 2019 |
| Versão identificada | `CheXpert-v1.0` no nome do ZIP e nos caminhos exibidos; revisão exata dos CSVs ainda não fixada por hash |
| Pacote efetivamente examinado | Dois arquivos completos de rótulos de treino conforme baixados; 223.414 registros em cada um. O conteúdo integral do pacote de imagens não foi recebido nem verificado |
| Distribuidor | Portal Stanford AIMI, conforme página apresentada pela pesquisadora |
| Página de referência | https://stanfordmlgroup.github.io/competitions/chexpert/ |
| Acesso ao pacote registrado | 14/10/2026; revisão documental em 16/10/2026 |
| URL exata da página de download | Pendente de registro no notebook; não inferida apenas da captura |
| CSV A | `train_cheXbert.csv`, tamanho anunciado de 23,13 MB; identificador desta ficha: **A — CheXbert** |
| CSV B | `train_visualCheXbert.csv`, tamanho anunciado de 29,86 MB; identificador desta ficha: **B — VisualCheXbert** |
| Outros componentes observados no portal | `README.md` e `CheXpert-v1.0.zip` de 471,12 GB |
| Hash SHA-256 e tamanho local exato dos CSVs | Não calculados nas saídas fornecidas |
| Inventário de imagens | Não executado; o ZIP não foi baixado nesta auditoria |
| `train.csv` do labeler CheXpert original | Não analisado; não confundir com o CSV A |
| `valid.csv` | Descrito no README, mas não obtido; não listado individualmente na captura. Conteúdo do ZIP não inspecionado |
| Papel proposto | Candidata à base de desenvolvimento/treinamento; papel definitivo depende das verificações e do protocolo |
| Notebook de origem | `01_auditoria_chexpert.ipynb`, executado localmente no VS Code |
| Histórico do projeto | Uso de CheXpert em 2025 informado por Aline; pacote, filtros, rótulos e transformações daquela versão ainda não conferidos |

Os arquivos A e B descrevem os mesmos caminhos de imagens. Constituem alternativas de anotação da mesma coleção de treino; não são duas bases independentes para avaliar generalização. CheXpert Plus deve receber registro próprio e verificação de sobreposição com CheXpert.

### Legenda de evidências

- **Documentada (D):** informação do README, portal, artigo ou repositório identificado. Não equivale a uma verificação local.
- **Verificada nos arquivos por saídas registradas (V):** resultado enviado pela pesquisadora após execução no notebook. Para reprodução independente, ainda é preciso conservar os arquivos, hashes e notebook com saídas.
- **Hipótese (H):** interpretação plausível, ainda sem teste que a confirme.
- **Não verificada (NV):** informação ou teste que permanece pendente.

### Registro de evidências da execução

Os códigos abaixo identificam evidências deste relatório; não são números de células do notebook.

| Código | Evidência disponível |
|---|---|
| E01 | Saídas de `shape`, lista de colunas e primeiras linhas dos dois CSVs |
| E02 | Contagens de `Pneumonia`, incluindo valores ausentes, para A e B |
| E03 | Tabela cruzada CheXbert × VisualCheXbert e cálculo de concordância |
| E04 | Saídas de idade, sexo, visão e projeção |
| E05 | Saída de 64.540 pacientes únicos, zero caminhos repetidos e igualdade de `Path` entre arquivos |
| E06 | README do pacote transcrito pela pesquisadora, com descrição dos labelers e da validação |
| E07 | Captura `image(9).png`: listagem de arquivos, tamanhos anunciados e indicação de termo aceito |
| E08 | Saídas de `df -h`, `free -h` e instalações do ambiente local |
| E09 | Relato de uso de CheXpert na versão de 2025; sem inspeção dos arquivos daquela versão |

## 3. Respostas às 24 perguntas essenciais

### 01. Quem coletou os dados, em quais instituições, país e período? Quem distribuiu este pacote?

- **Resposta e estado — D:** exames coletados retrospectivamente em Stanford Hospital, Estados Unidos, entre outubro de 2002 e julho de 2017, incluindo atendimento hospitalar e ambulatorial. Distribuição observada pelo portal Stanford AIMI.
- **Evidência:** [R1], E06 e E07.
- **Escopo e denominador:** descrição da coorte de origem; não é contagem local por unidade de atendimento.
- **Consequência:** permite caracterizar a origem institucional e temporal para a comparação de domínios.
- **Pendência:** registrar a URL exata e a revisão do pacote baixado. A composição por serviço clínico não foi medida nos CSVs.

### 02. Qual artigo, documentação, versão e licença/termo correspondem exatamente ao pacote?

- **Resposta e estado — D parcial:** a família CheXpert é descrita por Irvin et al. [R2]. A e B usam anotadores distintos, descritos por Smit et al. [R3] e Jain et al. [R5]. A captura indica o termo “Stanford University Dataset Research Use Agreement” e sua aceitação.
- **Evidência:** [R1–R6], E06 e E07.
- **Escopo e denominador:** um pacote de origem e dois arquivos de anotação; não há hash dos arquivos nem versão exata dos checkpoints usados em sua produção.
- **Consequência:** ano do dataset, versão do pacote e versão do anotador precisam de registros distintos.
- **Pendência:** arquivar o termo aplicável, README, URL, tamanhos locais e hashes. A captura não substitui o texto integral dos termos.

### 03. O pacote é completo ou uma amostra? Como ocorreu a seleção? Houve exclusões?

- **Resposta e estado — V/D parcial:** foram lidos integralmente os dois CSVs de treino baixados, com 223.414 linhas cada. Isso não confirma a recepção de todas as imagens ou dos conjuntos de validação e teste. Não foi realizada subamostragem local para essas contagens.
- **Evidência:** E01, E06 e E07.
- **Escopo e denominador:** 223.414 registros por arquivo; nenhum inventário de imagens.
- **Consequência:** a conclusão se limita às tabelas recebidas. Quantidade de linhas não comprova completude do pacote de imagens ou representatividade clínica.
- **Pendência:** documentar critérios de seleção/exclusão da coorte e confrontar o inventário recebido com a versão oficial quando as imagens estiverem acessíveis.

### 04. Quantas imagens, estudos e pacientes há na documentação e nos arquivos? Quanto ocupam?

- **Resposta e estado — D/V parcial:** a fonte anuncia 224.316 radiografias de 65.240 pacientes para CheXpert [R1]. Os CSVs de treino analisados contêm 223.414 registros por arquivo, 19 colunas e 64.540 identificadores de paciente extraídos dos caminhos. O número de estudos não foi calculado. Os tamanhos anunciados estão na seção 2.
- **Evidência:** [R1], E01, E05 e E07.
- **Escopo e denominador:** totais documentais da base e totais observados do treino são escopos diferentes; não somar A e B como novas imagens.
- **Consequência:** 223.414 linhas não representam 223.414 pacientes independentes.
- **Pendência:** contar estudos pela chave composta paciente/estudo, validar extração dos IDs e medir o tamanho real dos arquivos. Reconciliar os demais splits ao obtê-los.

### 05. Qual população está incluída: idades, sexo, contexto de atendimento, adultos/crianças?

- **Resposta e estado — V/D parcial:** idade média de 60,53 anos, mediana 62, mínimo 0 e máximo 110. Foram reportados 132.636 registros `Male`, 90.777 `Female` e 1 `Unknown`. A documentação inclui atendimento hospitalar e ambulatorial [R1].
- **Evidência:** E04; estatísticas completas na seção 4.3.
- **Escopo e denominador:** 223.414 linhas/imagens referenciadas; a distribuição pondera mais pacientes com mais exames. Não é distribuição por paciente único.
- **Consequência:** os valores de idade impedem assumir, apenas pelo resumo, que o conjunto observado seja exclusivamente adulto. O valor zero precisa de esclarecimento.
- **Pendência:** contar faixas etárias, examinar idades 0 e acima de 100 e caracterizar a população elegível. Preservar valores brutos; não excluir ou corrigir automaticamente.

### 06. Quais imagens foram fornecidas? Houve conversão, recorte, compressão ou redução?

- **Resposta e estado — V para metadados; NV para imagens:** os CSVs informam 191.027 registros frontais e 32.387 laterais, além das projeções da seção 4.3. Formato efetivo, resolução, profundidade, compressão e histórico de transformação das imagens não foram inspecionados.
- **Evidência:** E04 e E07.
- **Escopo e denominador:** visão/projeção de 223.414 registros; zero imagens examinadas nesta auditoria.
- **Consequência:** não há base para atribuir características de DICOM original, JPEG ou versão reduzida aos pixels ainda não recebidos.
- **Pendência:** fixar qual versão de imagens será usada, inspecionar formato e dimensões e documentar transformações. Conferir se o pacote usado em 2025 corresponde à versão escolhida agora.

### 07. Quais tarefas e rótulos existem? A que unidade pertence a anotação?

- **Resposta e estado — V/D parcial:** há 14 colunas de observações (seção 4.1). Os arquivos têm uma linha por caminho de imagem, enquanto os labelers recebem laudos como entrada. A referência textual pode corresponder a um estudo com várias imagens.
- **Evidência:** E01, E06, [R3–R6].
- **Escopo e denominador:** 223.414 registros por CSV; a consistência entre vistas do mesmo estudo não foi medida.
- **Consequência:** uma linha por imagem não comprova anotação independente de cada vista. Classificação multilabel é diferente de diagnóstico binário clínico.
- **Pendência:** identificar estudos, verificar repetição dos rótulos entre vistas e definir unidade de treino, avaliação e agregação. Caixas delimitadoras não fazem parte dos CSVs examinados.

### 08. O alvo “pneumonia” significa diagnóstico confirmado, suspeita no laudo, achado ou saída de modelo?

- **Resposta e estado — D/V:** em A, `Pneumonia` é a categoria inferida do texto pelo CheXbert. Em B, é uma previsão binária produzida pelo VisualCheXbert a partir do texto, visando correspondência com rótulos de imagem. Ambos são rótulos automáticos.
- **Evidência:** E02, E06, [R3–R6].
- **Escopo e denominador:** 223.414 registros por método; não há confirmação clínica individual fornecida nesses arquivos.
- **Consequência:** um futuro modelo será supervisionado por uma definição operacional baseada nessa referência. Rótulos 0 não significam automaticamente exame normal.
- **Pendência:** escolher e justificar a referência e seu uso; não adotar o método que fornece mais positivos apenas por ampliar a amostra.

### 09. Quais evidências originaram o rótulo? Elas foram disponibilizadas?

- **Resposta e estado — D/V parcial:** o README identifica laudos como entrada dos dois labelers. O treinamento de VisualCheXbert utiliza supervisão fornecida por um modelo de visão [R5], mas sua entrada para rotular é textual. Os CSVs recebidos apresentam rótulos e metadados, sem o texto dos laudos.
- **Evidência:** E01, E06 e [R5].
- **Escopo e denominador:** dois CSVs; ausência de colunas de laudo na lista de 19 colunas apresentada.
- **Consequência:** não é possível revisar a frase que originou cada rótulo somente com esses arquivos.
- **Pendência:** se a investigação exigir revisão textual, verificar uma fonte autorizada com vínculo aos mesmos estudos e documentar sua correspondência.

### 10. Há laudos no pacote? São integrais, seções ou texto processado? Em qual idioma?

- **Resposta e estado — V/NV:** não há coluna de laudo nos dois CSVs analisados. O código dos labelers usa a entrada denominada `Report Impression` [R4, R6]. A seção exata, seu processamento e idioma não foram verificados nos textos que geraram os arquivos recebidos, pois esses textos não estão disponíveis neste escopo.
- **Evidência:** E01 e documentação de uso dos anotadores.
- **Escopo e denominador:** inspeção do esquema dos dois CSVs; nenhuma leitura de laudos originais.
- **Consequência:** não transferir a disponibilidade de texto do CheXpert Plus automaticamente para este pacote.
- **Pendência:** registrar seção e preparação textual efetivamente utilizadas na geração de A e B, quando houver documentação específica.

### 11. Quem anotou manualmente, com qual qualificação, quantos leitores e como resolveu discordâncias?

- **Resposta e estado — D para validação; NV para revisão do treino:** o README descreve três radiologistas certificados anotando os estudos de validação; a referência resulta de votação após binarizar presença/incerteza provável como positivo e ausência/incerteza improvável como negativo. Isso descreve `valid.csv`, não os CSVs A e B.
- **Evidência:** E06. A documentação da competição também descreve referência radiológica [R1].
- **Escopo e denominador:** procedimento documental da validação; arquivo e contagens por classe não examinados.
- **Consequência:** não atribuir revisão humana individual aos 223.414 registros de treino. Referência radiológica também precisa ter tarefa e unidade explicitadas.
- **Pendência:** obter a referência pertinente, conferir cobertura de pneumonia e verificar qualquer revisão humana posterior dos CSVs de treino; atualmente não demonstrada.

### 12. Qual algoritmo/modelo/versão gerou a anotação automática, com qual entrada e treinamento?

- **Resposta e estado — D parcial:** A corresponde ao CheXbert, modelo de texto baseado em BERT treinado com anotações automáticas e ajuste com anotações especializadas [R3]. B corresponde ao VisualCheXbert, que recebe texto e aprende a aproximar rótulos de imagem com supervisão de um modelo visual [R5]. São métodos distintos do labeler CheXpert original baseado em regras.
- **Evidência:** E06 e [R3–R6].
- **Escopo e denominador:** método documentado de cada CSV. O repositório CheXbert informa que o checkpoint público foi treinado com dados MIMIC-CXR e difere do checkpoint estudado no artigo, treinado com CheXpert [R4].
- **Consequência:** nome do anotador e citação do artigo não identificam completamente a versão que gerou o arquivo.
- **Pendência:** fixar checkpoint/commit, processamento e limiares efetivos de A e B. Não há evidência local de revisão humana posterior de todas as saídas.

### 13. Como o anotador foi validado? Há avaliação independente e resultados específicos para pneumonia?

- **Resposta e estado — D parcial:** o artigo CheXbert descreve avaliação em laudos MIMIC-CXR anotados por especialistas, separados do desenvolvimento em CheXpert, com resultados por condição e análise de erros [R3]. Essa avaliação é de rotulagem textual. Para VisualCheXbert, o repositório informa que as métricas principais abrangem oito condições devido ao desbalanceamento do teste; pneumonia não está entre elas [R6].
- **Evidência:** [R3], seção de avaliação/tabela 2; [R6], nota sobre condições avaliadas.
- **Escopo e denominador:** estudos publicados, não validação independente dos 223.414 registros recebidos. O checkpoint público CheXbert difere do artigo, como registrado na pergunta 12.
- **Consequência:** não transportar uma métrica publicada ou a concordância de 79,1% para “acurácia de pneumonia” deste pacote. A existência da coluna em B não demonstra validação suficiente para essa classe.
- **Pendência:** documentar métricas, referência, classe, checkpoint e população pertinentes ao arquivo escolhido. Se necessário, avaliar uma amostra com referência independente adequada.

### 14. É possível identificar a origem de cada anotação e contar casos manuais, automáticos e desconhecidos?

- **Resposta e estado — D/V parcial:** a origem é identificada pelo arquivo: 223.414 registros associados a CheXbert e os mesmos 223.414 a VisualCheXbert. Não há coluna de revisão manual por registro entre as 19 colunas listadas.
- **Evidência:** E01, E05 e E06.
- **Escopo e denominador:** duas versões de anotação para uma mesma coleção de caminhos; não são 446.828 imagens independentes.
- **Consequência:** manter o nome da fonte dos rótulos no manifesto de qualquer subconjunto derivado.
- **Pendência:** não foi possível contar revisões humanas por linha; sua ausência no esquema não prova que nunca tenham ocorrido. Anotações especializadas de validação precisam ser registradas separadamente.

### 15. O que significam positivo, negativo, normal, incerto, não mencionado e ausente? Existem conflitos?

- **Resposta e estado — D/V:** A usa 1, 0, −1 e vazio para positivo, negativo, incerto e não mencionado. B usa previsão binária 1/0. As distribuições e divergências de pneumonia estão na seção 4.2. Ausência de menção é distinta de rótulo negativo; “normal” não é consequência de `Pneumonia=0`.
- **Evidência:** E02, E03, E06 e [R4, R6].
- **Escopo e denominador:** pneumonia em 223.414 registros; análise de concordância restrita a 8.170 com rótulo 0/1 em A.
- **Consequência:** manter incerto e não mencionado identificáveis durante a auditoria. Divergência entre métodos não identifica automaticamente qual anotação está errada.
- **Pendência:** validar valores permitidos e ausências nas 14 observações, estudar coocorrências e justificar qualquer conversão no protocolo. Não generalizar a completude de pneumonia em B às outras colunas sem verificar.

### 16. Existem achados relacionados e qual mapeamento seria necessário?

- **Resposta e estado — V:** `Lung Opacity` e `Consolidation`, entre outros, coexistem com `Pneumonia`. Sua presença no esquema não permite substituir uma coluna pela outra.
- **Evidência:** E01; lista completa na seção 4.1.
- **Escopo e denominador:** existência das colunas; coocorrências ainda não contadas.
- **Consequência:** comparar com a RSNA exige uma definição explícita, pois o alvo de opacidade da RSNA e os rótulos de pneumonia deste pacote têm referências diferentes.
- **Pendência:** analisar coocorrências e decidir se o desfecho permite comparação direta, tarefa harmonizada ou análises separadas. Registrar a informação perdida por qualquer mapeamento.

### 17. Quantos casos e pacientes permanecem após os critérios? Há positivos e negativos defensáveis?

- **Resposta e estado — V parcial:** um cenário descritivo que retém apenas `Pneumonia` 0/1 em A contém 8.170 registros: 4.856 positivos e 3.314 negativos. Em B há 52.226 positivos e 171.188 negativos. Não foram aplicados critérios conjuntos de idade, visão, projeção, disponibilidade de imagem ou independência.
- **Evidência:** E02 e E03.
- **Escopo e denominador:** registros/imagens referenciadas, não pacientes elegíveis. O número final de pacientes e estudos é desconhecido.
- **Consequência:** os 8.170 constituem um cenário de seleção por rótulo explícito, sujeito a viés de seleção; não são uma coorte final validada. Os negativos disponíveis são referências automáticas.
- **Pendência:** definir cenários de elegibilidade antes de olhar desempenho e contar imagens, estudos e pacientes em cada cenário.

### 18. IDs são estáveis e completos? Há várias imagens por estudo e exames repetidos?

- **Resposta e estado — V parcial:** os caminhos exibem componentes de paciente, estudo e vista; a extração realizada retornou 64.540 pacientes. A quantidade de registros é superior à de pacientes. A distribuição de estudos/imagens por paciente não foi calculada.
- **Evidência:** E01 e E05.
- **Escopo e denominador:** 223.414 registros de treino; estabilidade entre versões e cobertura completa da extração não demonstradas.
- **Consequência:** agrupar estudos com a chave paciente + estudo. Um identificador como `study1`, isolado, não identifica um estudo globalmente.
- **Pendência:** validar os caminhos, contar chaves compostas, múltiplas vistas e exames repetidos, e verificar consistência de rótulos por estudo. Definir agrupamento por paciente para divisões e incerteza estatística.

### 19. Imagens e metadados correspondem? Há arquivos ausentes, ilegíveis ou duplicados?

- **Resposta e estado — V para tabelas; NV para imagens:** foi reportado zero caminho repetido e igualdade exata da série `Path` entre A e B. Nenhum arquivo de imagem foi aberto nesta auditoria.
- **Evidência:** E05.
- **Escopo e denominador:** 223.414 linhas por CSV; zero imagens verificadas.
- **Consequência:** igualdade de caminhos entre tabelas não prova existência, legibilidade ou unicidade visual dos arquivos.
- **Pendência:** conferir `Path` ausente, fazer junção um a um por chave e comparar metadados entre A/B. Com as imagens, executar inventário, correspondências em ambos os sentidos, leitura/decodificação e identificação de duplicatas exatas e possíveis quase duplicatas, com revisão dos achados.

### 20. Quais metadados realmente existem, qual a ausência e qual sua confiabilidade?

- **Resposta e estado — V parcial:** os cinco campos descritivos são `Path`, `Sex`, `Age`, `Frontal/Lateral` e `AP/PA`. Idade não apresentou ausências após a conversão; sexo inclui um `Unknown`; `AP/PA` tem 32.387 ausências. Fabricante e parâmetros de exposição não integram o esquema recebido.
- **Evidência:** E01 e E04; seção 4.3.
- **Escopo e denominador:** 223.414 registros; não foi auditada a origem de cada campo nem o conteúdo de cabeçalhos das imagens.
- **Consequência:** “não disponível nos CSVs” não demonstra ausência desses dados na aquisição hospitalar ou em outra distribuição.
- **Pendência:** completar perfil de ausências e valores permitidos, rastrear origem dos campos e investigar idades, `Unknown`, `LL`/`RL` e coerência entre visão/projeção. Testar a hipótese de que as ausências de projeção correspondem às laterais.

### 21. Classe se associa a projeção, hospital, equipamento, idade, sexo, dispositivos ou anotador?

- **Resposta e estado — V parcial/NV:** foi observada grande diferença nas contagens de pneumonia por anotador e discordância em registros comuns. Os cruzamentos de rótulo com idade, sexo, visão, projeção e `Support Devices` ainda não foram executados. Hospital/equipamento por linha não estão no esquema recebido.
- **Evidência:** E02–E04.
- **Escopo e denominador:** comparações por método em 223.414 registros; nenhuma estimativa local de associação com os demais fatores.
- **Consequência:** distribuições gerais de AP/PA e idade não substituem análise de composição por classe. Não inferir confundimento ou causalidade apenas desses totais.
- **Pendência:** produzir cruzamentos com denominadores por estrato e incluir separadamente incerto/não mencionado; caracterizar o subconjunto realmente elegível.

### 22. Há splits oficiais? Como foram separados? Existe sobreposição com bases ou pré-treinamento?

- **Resposta e estado — D/NV:** a documentação distingue treino, validação e teste; o projeto descreve pacientes não vistos no teste [R1]. Nesta execução foram analisadas apenas as duas tabelas de treino. A/B compartilham caminhos; a ficha comum exige tratar CheXpert e CheXpert Plus como potencialmente sobrepostos.
- **Evidência:** [R1], E05, E06 e ficha padrão.
- **Escopo e denominador:** não foi calculada interseção de pacientes entre splits nem interseção entre bases.
- **Consequência:** referência ao split oficial não substitui conferir a partição recebida. Diferentes labelers ou distribuições da mesma origem não criam teste externo independente.
- **Pendência:** obter os índices necessários, medir interseções de pacientes/imagens, fixar divisões por paciente e verificar dados de pré-treinamento dos modelos escolhidos. Não reutilizar um teste reservado para ajustar escolhas do modelo.

### 23. Quais estatísticas de pixels podem ser medidas e como interpretá-las?

- **Resposta e estado — NV:** nenhuma estatística de pixels foi calculada no CheXpert nesta execução. A comparação inicial com a RSNA deverá explicitar formato, escala, profundidade, transformações e método de amostragem.
- **Evidência:** delimitação do escopo recebido; E07.
- **Escopo e denominador:** zero imagens analisadas; não há resultados a preencher.
- **Consequência:** não há evidência atual de diferenças de brilho, contraste, qualidade ou exposição física nesta base.
- **Pendência:** selecionar amostra estratificada com semente registrada, medir descritores comparáveis (média, dispersão, percentis e extremos) e fazer inspeção visual sistemática. Registrar fundo/margens e transformações; evitar comparar diretamente intensidades brutas de escalas incompatíveis. O tamanho e a seleção da amostra devem ser documentados.

### 24. Qual papel é defensável, com quais restrições e verificações obrigatórias?

- **Resposta e estado — parecer baseado em D/V e pendências:** manter CheXpert como candidata à base de desenvolvimento. Há rótulos de interesse, identificadores de paciente e metadados úteis nos CSVs. A disponibilidade e integridade das imagens, a referência de pneumonia e a coorte elegível ainda precisam de verificação.
- **Evidência:** E01–E07 e perguntas 01–23. E09 registra experiência anterior, sem substituir validação dos arquivos atuais.
- **Escopo e denominador:** dois CSVs sobre 223.414 registros compartilhados; 64.540 pacientes identificados nos caminhos; estudos ainda não contados; nenhuma imagem inspecionada.
- **Consequência:** evidência insuficiente para aprovação definitiva do pacote como base de desenvolvimento ou teste clínico. Não foi medida contribuição percentual, desempenho ou generalização.
- **Pendência:** executar o plano da seção 5 e atualizar o parecer depois de registrar as novas saídas.

## 4. Resultados preservados da execução

### 4.1. Estrutura

| Verificação | Resultado registrado |
|---|---:|
| Formato de A | 223.414 × 19 |
| Formato de B | 223.414 × 19 |
| Pacientes únicos extraídos dos caminhos | 64.540 |
| Caminhos repetidos, conforme saída apresentada | 0 |
| Séries `Path` de A e B iguais e na mesma ordem | Sim |

**Campos descritivos:** `Path`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA`.

**Observações:** `Enlarged Cardiomediastinum`, `Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`, `Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`, `Fracture`, `Support Devices`, `No Finding`.

A igualdade reportada refere-se a `Path`; não houve demonstração de igualdade de todas as demais colunas de metadados.

### 4.2. Pneumonia: distribuição e comparação

Denominador das distribuições: **223.414 registros por arquivo**.

| Categoria | A — CheXbert | B — VisualCheXbert |
|---|---:|---:|
| 1 — positivo do anotador | 4.856 | 52.226 |
| 0 — negativo do anotador | 3.314 | 171.188 |
| −1 — incerto | 19.386 | 0 observado |
| Vazio/NaN | 195.858 — não mencionado | 0 observado |
| Total | 223.414 | 223.414 |

A ausência de valores vazios foi observada para pneumonia em B; não foi demonstrada para todas as demais observações.

| A — CheXbert | B: 0 | B: 1 | Total |
|---|---:|---:|---:|
| −1 — incerto | 9.954 | 9.432 | 19.386 |
| 0 — negativo | 2.959 | 355 | 3.314 |
| 1 — positivo | 1.352 | 3.504 | 4.856 |
| Não mencionado | 156.923 | 38.935 | 195.858 |
| Total | 171.188 | 52.226 | 223.414 |

- Registros com A explicitamente binário: **8.170**.
- Rótulos iguais nesses registros: **6.463**.
- Concordância descritiva: **6.463 / 8.170 = 79,1%**.
- Discordâncias nesse subconjunto: **1.707**.
- Entre A=0: B também é 0 em **2.959/3.314 = 89,3%**.
- Entre A=1: B também é 1 em **3.504/4.856 = 72,2%**.
- Entre os 195.858 não mencionados em A, B atribui 1 a **38.935 (19,9%)**.

**Limite da comparação:** são dois anotadores, sem referência independente adicionada a esta análise. Não chamar essas discordâncias de falsos positivos ou falsos negativos. As diferenças envolvem métodos e objetivos de anotação distintos; não foram demonstradas como simples substituição de valores incertos/vazios. A concordância bruta tampouco demonstra acurácia ou utilidade percentual do dataset.

### 4.3. Metadados: contagens por registro de imagem

| Idade | Resultado |
|---|---:|
| Contagem válida na saída | 223.414 |
| Média | 60,527662 anos |
| Desvio-padrão | 17,995927 anos |
| Mínimo | 0 |
| Primeiro quartil | 49 |
| Mediana | 62 |
| Terceiro quartil | 74 |
| Máximo | 110 |
| Ausentes após conversão numérica | 0 |
| Negativas | 0 |
| Acima de 100 | 140 |

Os 140 registros acima de 100 correspondem a aproximadamente 0,063% das linhas. São valores a investigar, sem classificação automática como erro. A idade zero também demanda esclarecimento. Não foram produzidas contagens de pacientes únicos por idade ou faixa etária.

| Campo | Categoria | Contagem |
|---|---|---:|
| Sexo | Male | 132.636 |
| Sexo | Female | 90.777 |
| Sexo | Unknown | 1 |
| Visão | Frontal | 191.027 |
| Visão | Lateral | 32.387 |
| Projeção | AP | 161.590 |
| Projeção | PA | 29.420 |
| Projeção | LL | 16 |
| Projeção | RL | 1 |
| Projeção | NaN | 32.387 |

**Hipótese a testar:** o número de projeções ausentes é igual ao número de laterais. A igualdade dos totais não comprova que sejam as mesmas linhas; é necessário cruzar os campos. `LL` e `RL` no campo denominado `AP/PA` também precisam ser confrontados com a visão e a documentação, sem remapeamento automático.

## 5. Plano para completar a auditoria comum

### 5.1. Verificações possíveis com os CSVs atuais

| ID | Verificação pendente | Evidência esperada |
|---|---|---|
| C01 | Registrar URL, versão, tamanhos, hashes SHA-256 e notebook com saídas | Manifesto dos dois arquivos e registro reproduzível da execução |
| C02 | Validar esquema, valores permitidos das 14 observações, campos ausentes, `Path` vazio/repetido e igualdade dos metadados entre A/B | Tabelas de contagens, divergências e junção por caminho com cardinalidade verificada |
| C03 | Extrair paciente e estudo, validar cobertura, contar imagens por estudo e estudos por paciente | Totais de imagens/estudos/pacientes e casos que exigem agrupamento ou investigação |
| C04 | Cruzar rótulos de pneumonia com visão, projeção, sexo, idade e dispositivos em cada arquivo | Contagens e proporções com denominadores, preservando incerto/não mencionado e estratos ausentes |
| C05 | Investigar idade, `Unknown`, `LL`/`RL` e coerência de visão/projeção | Lista de ocorrências e hipóteses testadas, com valores brutos preservados |
| C06 | Quantificar cenários de elegibilidade definidos para o projeto | Imagens, estudos e pacientes positivos/negativos por cenário, sem escolha baseada em desempenho |

### 5.2. Verificações que exigem imagens ou outras referências

| ID | Verificação pendente | Evidência esperada |
|---|---|---|
| I01 | Identificar pacote usado em 2025 e/ou planejar acesso oficial à versão escolhida | Fonte, transformações e capacidade de armazenamento confirmadas |
| I02 | Inventariar imagens e reconciliar caminhos em ambos os sentidos | Quantidade recebida, arquivos sem rótulo e rótulos sem arquivo |
| I03 | Ler/decodificar imagens do pacote escolhido e medir formato, dimensões, profundidade e falhas | Cobertura real do teste e lista de falhas; uma amostra deve ser identificada como amostra |
| I04 | Analisar pixels em amostra estratificada e realizar inspeção visual sistemática | Semente, seleção, denominadores, estatísticas e achados visuais rastreáveis |
| I05 | Investigar duplicatas exatas e possíveis quase duplicatas | Resultados por método, revisão dos candidatos e impactos no agrupamento |
| P01 | Obter índices dos splits/referências pertinentes e conferir pacientes e sobreposição | Interseções entre splits e bases, incluindo CheXpert Plus; divisão por paciente antes de treinar |
| P02 | Documentar anotador, checkpoint, significado do alvo e política de incerteza | Referência de treino e avaliação definida; limitações da validação específica para pneumonia |
| P03 | Conferir dados de pré-treinamento do modelo escolhido | Registro de possível exposição prévia ao conjunto de avaliação |

Os rótulos especializados de validação/teste só podem sustentar avaliação após verificar disponibilidade, tarefa, contagens e independência. A falta de `valid.csv` isolado no portal não prova inexistência do arquivo ou necessidade inevitável de baixar o ZIP inteiro.

### 5.3. Recursos computacionais informados

O computador apresentou aproximadamente **30 GiB de RAM** e **866 GiB livres no volume de dados** nas saídas de 14/10/2026 (E08). O ZIP anunciado é de **471,12 GB**. Essas unidades e a data da medição devem ser preservadas: ainda faltam tamanho descompactado, espaço temporário e espaço para resultados. GPU e memória de vídeo não foram verificadas. O ambiente de notebook foi funcional para os CSVs; isso não confirma capacidade de treinamento de imagens.

### 5.4. Relação com a auditoria RSNA

| Eixo comum | Situação atual no CheXpert | Complementação necessária |
|---|---|---|
| Proveniência e tamanho | Fonte e dois CSVs identificados | Fixar versão/hashes e inventariar imagens |
| Rótulos e anotações | Pneumonia comparada entre dois métodos | Demais validações estruturais, unidade de anotação e referência escolhida |
| Integridade técnica | Integridade tabular parcial | Correspondência e leitura real das imagens |
| Composição | Distribuições gerais | Cruzamentos por classe e coorte elegível |
| Pixels | Não executado | Amostra estratificada e interpretação das transformações |
| Dependência | Contagem de pacientes pelos caminhos | Estudos, repetições, splits e sobreposição |
| Parecer | Manter como candidata | Aprovação do papel experimental após resolver pendências determinantes |

Mesmo nível de exigência significa documentar o que foi efetivamente testado em cada eixo. As caixas geométricas da RSNA não exigem teste equivalente no CheXpert, que não as fornece nos arquivos analisados. A ausência de uma verificação comum deve permanecer visível; a existência de documentação não deve ser marcada como leitura de imagem executada.

## 6. Parecer padronizado

> Auditamos os CSVs `train_cheXbert.csv` e `train_visualCheXbert.csv` distribuídos para CheXpert-v1.0, abrangendo os mesmos 223.414 registros de imagem e 64.540 identificadores de paciente extraídos dos caminhos; o total de estudos ainda não foi contado. Verificamos a estrutura apresentada, os resultados de unicidade/igualdade dos caminhos, a distribuição e concordância de pneumonia e estatísticas demográficas e de visão/projeção. O alvo é uma anotação automática derivada de laudos, com método e significado diferentes entre os arquivos. O papel proposto é desenvolvimento/treinamento, condicionado à definição da referência, à confirmação das imagens e ao agrupamento por paciente. Permanecem não verificados a integridade real das imagens, seus pixels, a elegibilidade final, as associações entre classe e metadados e a independência entre partições/bases. A decisão é **manter como candidata**, porque os CSVs oferecem estrutura e informações relevantes para a tarefa; há **evidência insuficiente para aprovação definitiva do pacote no experimento**.

**A ficha foi preenchida com as evidências disponíveis; isso não significa que os 24 itens tenham sido integralmente verificados.** O relatório será atualizado após as execuções pendentes. O uso em 2025 é um antecedente do projeto, não uma validação dos arquivos atuais. Nenhuma conclusão sobre acurácia clínica, erro do modelo, calibração ou generalização foi produzida nesta etapa.

## 7. Fontes documentais e registros

Fontes externas consultadas na revisão em **16/10/2026**. Os artigos descrevem seus próprios experimentos; suas métricas não foram transferidas para os CSVs locais.

- **[R1]** Stanford Machine Learning Group. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*. https://stanfordmlgroup.github.io/competitions/chexpert/
- **[R2]** Irvin J. et al. *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*. 2019. https://arxiv.org/abs/1901.07031
- **[R3]** Smit A. et al. *CheXbert: Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT*. 2020. https://arxiv.org/abs/2004.09167 — especialmente avaliação em MIMIC-CXR, tabela 2 e análise de erros.
- **[R4]** Stanford ML Group. Repositório oficial CheXbert, README: checkpoint público, entrada textual e convenção de rótulos. https://github.com/stanfordmlgroup/CheXbert
- **[R5]** Jain S. et al. *VisualCheXbert: Addressing the Discrepancy Between Radiology Report Labels and Image Labels*. 2021. https://arxiv.org/abs/2102.11467
- **[R6]** Stanford ML Group. Repositório oficial VisualCheXbert, README: entrada textual, saída binária e subconjunto de condições avaliadas. https://github.com/stanfordmlgroup/VisualCheXbert
- **[E01–E05]** Saídas de `01_auditoria_chexpert.ipynb` fornecidas pela pesquisadora nesta auditoria, execução local de 14/10/2026. Preservação do notebook completo e vinculação dos identificadores/células: pendentes.
- **[E06–E09]** README transcrito, captura da página de download, saídas do ambiente e histórico informados pela pesquisadora. Seus escopos estão descritos no registro de evidências.
- **Padrão metodológico:** `Ficha_cientifica_datasets_DIAMANTE.md`, versão 16/10/2026, fornecida pela pesquisadora.

## 8. Histórico de revisão

- **14/10/2026:** análise inicial dos CSVs e primeiro relatório preliminar.
- **16/10/2026:** reorganização segundo as 24 perguntas comuns; rastreabilidade e denominadores explicitados; distinção entre evidência documental e saídas registradas; inclusão das limitações específicas dos anotadores; status corrigido para auditoria padronizada em andamento e decisão de manter como candidata, sem aprovação integral do pacote.
