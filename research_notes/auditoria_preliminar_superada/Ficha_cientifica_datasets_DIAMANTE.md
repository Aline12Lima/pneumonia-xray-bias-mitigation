# DIAMANTE — Ficha científica padronizada dos datasets

Versão: 16/10/2026. Usar uma cópia por **pacote/versão** auditado. CheXpert, CheXpert Plus e diferentes arquivos de rótulos precisam de registros identificáveis, mesmo quando compartilham imagens.

## Objetivo

Responder: **o pacote disponível é íntegro e adequado ao papel experimental proposto, e quais limites precisam entrar no protocolo?**

Papéis candidatos: desenvolvimento; avaliação interna; avaliação externa; extensão pediátrica. A decisão depende da compatibilidade dos rótulos, dos pacientes e das imagens efetivamente disponíveis. O tamanho da base sozinho não determina seu papel.

## Como registrar as respostas

Para cada pergunta, preencher: **resposta | evidência (link, arquivo ou célula) | escopo e denominador | consequência para o projeto | pendência**.

Marcar a evidência como **documentada**, **verificada nos arquivos**, **hipótese** ou **não verificada**. Documentação e verificação podem coexistir. Usar “não disponível no pacote” para um campo ausente; isso não prova que a informação nunca existiu no hospital. Não preencher lacunas com suposições.

Identificação da ficha:

- Dataset e versão:
- Pacote: completo / amostra / subconjunto / transformação de terceiros:
- Fonte e responsável pela distribuição:
- URL, data de acesso e versão do pacote:
- Hash do CSV e inventário das imagens:
- Papel proposto e justificativa inicial:
- Notebook e data de execução:
- Escopo efetivamente verificado:

## Perguntas essenciais

| Nº | Pergunta | Por que importa para a decisão científica? |
|---|---|---|
| 1 | Quem coletou os dados, em quais instituições, país e período? Quem distribuiu este pacote? | Distingue origem clínica de cópia/reempacotamento e caracteriza mudança de domínio. |
| 2 | Qual artigo, documentação, versão e licença/termo correspondem exatamente ao pacote? | Permite rastrear e reproduzir o experimento e conhecer condições de uso. |
| 3 | O pacote é completo ou uma amostra? Como ocorreu a seleção? Houve exclusão por qualidade, diagnóstico ou disponibilidade? | Identifica seleção potencialmente enviesada e limites da generalização. |
| 4 | Quantas imagens, estudos e pacientes há na documentação e nos arquivos? Quanto ocupam? | Distingue tamanho anunciado, tamanho recebido e unidades independentes. |
| 5 | Qual população está incluída: idades, sexo, contexto ambulatorial/internação, adultos/crianças? | Define elegibilidade e comparabilidade entre bases. |
| 6 | Quais imagens foram fornecidas: projeções, formato, resolução e profundidade? Houve conversão, recorte, compressão ou redução? | Define o que pode ser comparado e medido nos pixels. |
| 7 | Quais tarefas e rótulos existem? A anotação pertence à imagem, ao estudo, ao paciente, a uma região ou ao laudo? | Evita atribuir evidência a uma unidade diferente daquela que foi anotada. |
| 8 | O alvo “pneumonia” significa diagnóstico clínico confirmado, suspeita no laudo, achado radiológico ou saída de outro modelo? | Estabelece o que o modelo poderá afirmar e se os alvos são compatíveis. |
| 9 | Quais evidências originaram o rótulo: imagem, laudo, prontuário, exame laboratorial ou combinação? Essas evidências foram disponibilizadas? | Distingue origem da anotação de material disponível para auditá-la. |
| 10 | Há laudos no pacote? São integrais, seções, texto processado ou apenas rótulos extraídos? Em qual idioma e vinculados a quais IDs? | Determina quanto da anotação pode ser inspecionado e rastreado. |
| 11 | Quem anotou manualmente, com qual qualificação, quantos leitores, instruções e resolução de discordâncias? Eles anotaram imagens ou texto? | Caracteriza o padrão de referência; “manual” não equivale automaticamente a diagnóstico confirmado. |
| 12 | Se houve anotação automática, qual algoritmo/modelo/versão foi usado, com qual entrada e referência de treinamento? Houve revisão humana posterior? | Distingue extração de laudos de pseudo-rótulos gerados a partir das imagens. |
| 13 | Como o anotador automático foi validado? Em dados independentes? Há desempenho por classe, especialmente pneumonia, e análise de erros? | Uma métrica global de extração de texto não prova a precisão do rótulo de pneumonia. |
| 14 | É possível identificar a origem de cada anotação e contar casos manuais, automáticos, mistos e desconhecidos? | Permite escolher ou estratificar o conjunto por confiabilidade da referência. |
| 15 | O que significam positivo, negativo, normal, incerto, não mencionado e ausente? Existem conflitos ou multilabels? | Impede converter ausência de menção em ausência confirmada da doença. |
| 16 | Existem achados relacionados (opacidade, consolidação etc.)? Qual mapeamento seria necessário e o que ele perderia? | Evita equiparar conceitos apenas porque estão associados à pneumonia. |
| 17 | Quantos casos e pacientes permanecem após critérios de população, projeção e referência? Há positivos e negativos defensáveis? | Mede viabilidade do experimento, sem ajustar critérios ao desempenho do modelo. |
| 18 | IDs de imagem, estudo e paciente são estáveis e completos? Há várias imagens por estudo e exames repetidos? | Orienta agrupamento, divisão e cálculo de incerteza por paciente. |
| 19 | Imagens e metadados correspondem? Há arquivos ausentes, ilegíveis, IDs repetidos, imagens idênticas ou quase duplicadas? | Delimita integridade; cada teste detecta um tipo diferente de problema. |
| 20 | Quais metadados técnicos e demográficos realmente existem e qual a taxa de ausência? Qual a origem e confiabilidade de cada campo? | Determina quais fatores de aquisição podem ser estudados. |
| 21 | Classe se associa a projeção, hospital, equipamento, idade, sexo, dispositivos ou método de anotação? | Identifica confundimento e possíveis atalhos do modelo. Associação não comprova causalidade. |
| 22 | Há splits oficiais? Foram separados por paciente, tempo ou instituição? Existe sobreposição com outras bases ou pré-treinamento do modelo? | Evita vazamento e impede chamar um teste de externo sem verificar independência. |
| 23 | Quais estatísticas de pixels podem ser medidas e como a transformação das imagens afeta sua interpretação? | Evita interpretar intensidade digital como exposição física ou qualidade clínica. |
| 24 | Qual papel é defensável, com quais restrições, incertezas e verificações obrigatórias antes do uso? | Converte a auditoria em decisão documentada. |

## Entendendo a anotação automática

Um processo possível é: **radiologista interpreta a imagem → escreve um laudo → pessoas anotam parte dos laudos → um modelo de linguagem aprende essas anotações → o modelo extrai rótulos dos demais laudos**. Nesse caso, o rótulo automático vem do texto. Ele pode herdar tanto as limitações do laudo quanto os erros do extrator.

Outro processo possível é: **modelo recebe a imagem → produz um rótulo previsto**. Isso gera pseudo-rótulos baseados na imagem; a referência usada na avaliação precisa ser descrita como tal. Há também sistemas baseados em regras e processos híbridos. É necessário identificar qual foi usado em cada arquivo.

Ter rótulos sem receber os laudos não prova que os rótulos foram produzidos sem laudos. Ter laudos disponíveis não prova que todos os rótulos foram revisados por humanos.

## PadChest — informações documentais já identificadas

Estas informações descrevem a fonte, e não resultados da execução do notebook.

| Item | O que está documentado | O que ainda verificar no pacote |
|---|---|---|
| Origem | Hospital San Juan, Espanha; exames entre 2009 e 2017. [1, 3] | Identificação e versão da amostra efetivamente anexada. |
| Tamanho da base | Mais de 160 mil imagens e aproximadamente 67 mil pacientes; página oficial informa 1,02 TB. [1, 3] | Contagens da amostra separadas das contagens publicadas. |
| Acesso à amostra | O link de amostra do BIMCV direciona à página de raddar no Kaggle. [1, 4] | Inventário local, CSV, formatos e correspondências. |
| Referência | Rótulos de achados e diagnósticos diferenciais obtidos a partir de laudos. [2, 3] | Vocabulário e unidade dos rótulos; definição dos positivos e negativos do projeto. |
| Anotação | Aproximadamente 27% dos laudos anotados por médicos; demais por método supervisionado de texto, baseado em rede recorrente com atenção. [1, 3] | Valores e cobertura de `MethodLabel` na amostra; distribuição específica para pneumonia. |
| Texto fornecido | O repositório descreve `Report` como trecho de interpretação processado, com palavras reduzidas aos radicais e tokenizadas. [2] | Existência da coluna, preenchimento e vínculo por estudo no CSV recebido. |
| Métrica do anotador | Micro-F1 global de 0,93 na avaliação relatada. [3] | Não interpretar como acurácia de pneumonia, confiança de cada imagem ou desempenho do nosso modelo. |
| Metadados | Dicionário inclui IDs, projeções, informações demográficas, técnicas e método de anotação. [2] | Disponibilidade, consistência e valores ausentes por campo. |
| Representatividade | O fato de ser uma amostra disponibilizada não garante seleção aleatória ou representativa. | Método de seleção e número de casos elegíveis; registrar desconhecido se não houver evidência. |

## Perguntas adicionais por família

Estas são perguntas de verificação; não representam resultados novos de auditoria.

- **RSNA:** qual Stage foi recebido? O que `Target=0` reúne? As várias linhas são caixas diferentes da mesma imagem? Qual é o resultado do mapeamento de imagens para pacientes de origem NIH? O teste da competição possui rótulos acessíveis para a tarefa proposta?
- **CheXpert:** os rótulos são do CheXpert original, CheXbert ou VisualCheXbert? Qual método e versão gerou cada CSV? O que significam 1, 0, −1 e vazio? Quais arquivos têm referência revisada por especialistas? A imagem reduzida preserva os fatores que pretendemos estudar?
- **CheXpert Plus:** quais imagens compartilha com CheXpert? Quais seções do laudo estão presentes? Qual seção e extrator geraram cada conjunto de rótulos? Como tratar divergências? A cobertura medida refere-se ao índice direto, aos arquivos compactados ou à leitura real das imagens?
- **PadChest:** quais casos têm anotações manuais versus automáticas? `Labels` e `Report` estão completos? Os rótulos do estudo foram associados a várias projeções? `Projection` foi recuperada do DICOM ou inferida? Quais parâmetros de aquisição realmente estão preenchidos?
- **Pediatria, por último:** qual dataset e versão? Qual faixa etária e critério diagnóstico? É possível agrupar por paciente? A mudança para crianças será tratada como experimento separado de mudança populacional?

**CheXpert e CheXpert Plus precisam ser tratados como fontes potencialmente sobrepostas.** Uma embalagem diferente não garante um teste independente: verificar IDs e origem antes de atribuir papéis separados.

## Mesmo roteiro de auditoria, com rastreabilidade

1. Proveniência e tamanho → identificação, versão e inventário.
2. Rótulos e anotações → cadeia de produção da referência e semântica do alvo.
3. Integridade técnica → correspondência, leitura, dimensões e metadados.
4. Composição → distribuição e cruzamentos relevantes.
5. Pixels → descrição exploratória com escopo e transformação registrados.
6. Dependência → pacientes, estudos, duplicações e sobreposição.
7. Parecer → papel, limites, evidências e pendências para o protocolo.

O núcleo é comum. As verificações adicionais dependem da base: caixas na RSNA; incerteza e origem dos rótulos em CheXpert; seções do laudo em Plus; origem manual/automática em PadChest.

## Critério para encerrar a etapa

Encerrar a auditoria preliminar quando houver informação suficiente para decidir a viabilidade e listar explicitamente as verificações restantes. Uma pendência que determina a comparabilidade dos rótulos ou a disponibilidade das imagens impede aprovação definitiva para o experimento. Uma amostra pode aprovar o procedimento de leitura sem aprovar a base completa.

Não atribuir uma porcentagem subjetiva de “contribuição”. Registrar uma decisão sustentada: **manter como candidata**, **manter com restrições**, **inviável para o papel proposto** ou **evidência insuficiente**.

Modelo de parecer:

> Auditamos [pacote/versão], abrangendo [imagens/estudos/pacientes]. Verificamos [testes]. O alvo corresponde a [definição e origem]. O papel proposto é [papel], condicionado a [restrições]. Permanecem não verificados [itens]. Esses limites afetam [conclusões permitidas]. A decisão é [parecer], porque [evidências].

## Fontes

1. BIMCV. PadChest: descrição, acesso e link da amostra. https://bimcv.cipf.es/bimcv-projects/padchest/ (consultado em 16/10/2026).
2. Bustos e colaboradores. Repositório e dicionário de campos. https://github.com/auriml/Rx-thorax-automatic-captioning (consultado em 16/10/2026).
3. Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M. *PadChest: A large chest x-ray image dataset with multi-label annotated reports*. Medical Image Analysis, 2020, 66:101797. https://doi.org/10.1016/j.media.2020.101797
4. Kaggle. PadChest Chest X-rays (sample). https://www.kaggle.com/datasets/raddar/padchest-chest-xrays-sample (página identificada pelo link do BIMCV).
