Qualidade de radiografias de tórax e classificadores de pneumonia

Pesquisa de iniciação científica que investiga como a qualidade e a apresentação de radiografias de tórax se relacionam com os erros (falsos positivos e falsos negativos), a calibração e a generalização entre hospitais de um classificador de pneumonia.

Status (27/10/2026): auditoria das três bases concluída, bases aprovadas. Em andamento: fechamento e congelamento do protocolo (previsto para 31/10). Nenhum modelo final foi treinado e o conjunto de teste continua trancado.

Pergunta de pesquisa

Em hospitais independentes, em que medida diferenças nas distribuições de características naturais de qualidade e de parâmetros de aquisição estão associadas à diferença de erro e de calibração de um classificador de pneumonia, após padronização de posição, idade e sexo?

Hipóteses
	Hipótese
H1	Algumas características de qualidade se associam a falsos positivos e falsos negativos, após ajuste por posição, idade e sexo.
H2	A diferença de desempenho entre hospitais diminui quando essas características são padronizadas.
H3	O efeito da qualidade sobre a calibração difere do efeito sobre a discriminação.
H4	O resultado pode ser nulo ou heterogêneo; o estudo não pressupõe que a qualidade seja a principal explicação.

A linguagem do estudo é de associação, não de causa.

Desenho

O modelo é mantido fixo para que as diferenças observadas venham dos dados, e não da arquitetura.

Arquitetura: DenseNet121 com pesos ImageNet (sem pesos treinados em radiografias, para preservar a independência dos testes externos).
Entrada: 320 × 320, 8 bits, sem equalização de histograma.
Aumento de dados: apenas geométrico leve; nenhum aumento fotométrico.
Sementes: 5 por modelo.
Modelo	Treino	Teste interno	Teste externo
M1	CheXpert Plus	CheXpert Plus	RSNA, PadChest
M2	RSNA	RSNA	CheXpert Plus, PadChest
M3	CheXpert Plus + RSNA (balanceado por fonte)	CheXpert Plus e RSNA	PadChest
Análises
Análise	Pergunta
Desempenho de referência	Qual é a queda entre teste interno e externo (AUROC, AUPRC, FNR, FPR, ECE)?
Observacional	A qualidade natural da imagem se associa aos erros e à calibração?
Interventiva	Degradações controladas (gama, contraste, brilho, desfoque, ruído, resolução) aumentam o erro de forma dose-resposta?
Explicação da queda externa	Quanto do gap diminui ao padronizar fatores clínicos e, depois, a qualidade?
Diagnóstico dos dados	A qualidade carrega informação do rótulo? Existe assinatura de origem?
Bases de dados

Três hospitais distintos, sem pacientes em comum.

Base	Origem	Papel	Imagens na base final	Positivos	Rótulo
CheXpert Plus	Stanford (EUA)	Treino (M1, M3) e teste	176.660	3.444	Laudo (CheXbert na impressão)
RSNA Pneumonia	NIH Clinical Center (EUA)	Treino (M2, M3) e teste	25.403	5.627	Radiologistas, por imagem
PadChest	Hospital San Juan, Alicante (Espanha)	Só teste externo	24.770	796	Médicos, por laudo

Filtros comuns: radiografias frontais AP/PA, uma por estudo, adultos de 18 a 100 anos, sexo registrado e sem duplicatas exatas.

Ficaram de fora: CheXpert em JPG (equalizado, apaga brilho e contraste), NIH completo (contém o RSNA), MIMIC-CXR e VinDr (credenciamento), bases pediátricas (versão futura).

Principais achados da auditoria
Nenhuma base traz a imagem crua: o estudo fala em imagens distribuídas.
A posição se associa à pneumonia em direções opostas entre bases (mais no AP no RSNA e no PadChest; mais no PA no CheXpert Plus).
Fabricante, qualidade e prevalência andam juntos no CheXpert Plus e no PadChest.
O arquivo de laudo completo do CheXpert Plus está desalinhado com as imagens; o rótulo vem só da impressão.
O mesmo modelo de máscara pulmonar (lungs-segmentation, treinado em Montgomery e Shenzhen) é usado nas três bases.

Detalhes em docs/.

Dados e termos de uso

Este repositório não contém imagens nem dados por imagem. As bases devem ser obtidas diretamente nas fontes oficiais, respeitando os termos de cada uma:

CheXpert Plus: uso só para pesquisa, sem uso comercial e sem redistribuição. Distribuído pela Stanford AIMI via Redivis.
RSNA Pneumonia Detection Challenge (2018): distribuição oficial da RSNA. Atribuição a Wang et al. (2017) e Shih et al. (2019).
PadChest: uso só acadêmico; redistribuição de qualquer parte é proibida. Acesso por solicitação ao BIMCV.

Só serão publicados código e resultados agregados (tabelas, métricas e figuras).

Infraestrutura
Código: este repositório, editado localmente.
Processamento e treino: notebooks no Redivis, onde o CheXpert Plus em DICOM (2,58 TB) está hospedado; as imagens são processadas em fluxo, sem download do pacote completo.
Rastreamento: cada execução salva um YAML de configuração junto da tabela de predições.
Estrutura do repositório
├── configs/            # configurações YAML por execução
├── data/               # fora do git (dados locais e cache)
│   └── splits/         # listas de IDs por partição, com hash SHA-256
├── docs/
│   ├── decisoes/       # registro de decisões
│   ├── desvios/        # desvios após o congelamento do protocolo
│   └── diario/         # diário de execução
├── results/
│   └── predicoes/      # tabelas de predição (base de todas as análises)
└── src/                # código-fonte
Governança
O protocolo é congelado antes do treino; depois disso, qualquer mudança é registrada como desvio, com data, motivo e impacto.
A divisão é por paciente (70/10/20, semente 2026); as listas de IDs são versionadas com hash.
O conjunto de teste só é aberto depois do treino final e do congelamento do código de análise.
Multiplicidade: correção de Holm nas análises primárias e Benjamini-Hochberg nas secundárias, com margens de relevância prática.
Cronograma
Etapa	Período	Situação
Delimitação e fundamentação	até 17/10/2026	Concluída
Infraestrutura	18 a 23/10	Concluída
Auditoria das bases	18 a 28/10	Bases aprovadas; faltam medidas de qualidade
Congelamento do protocolo	27 a 31/10	Em andamento
Modelo de referência	1 a 7/11	A fazer
Treino de M1, M2 e M3	8 a 20/11	A fazer
Abertura do teste	21/11	A fazer
Análises	21/11 a 12/12	A fazer
Reprodutibilidade e redação	8 a 31/12	A fazer
Referências principais
Zech JR et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. PLOS Medicine.
Chambon P, Delbrouck JB et al. (2024). CheXpert Plus: augmenting a large chest X-ray dataset with text radiology reports, patient demographics and additional image formats. arXiv:2405.19538.
Shih G et al. (2019). Augmenting the National Institutes of Health chest radiograph dataset with expert annotations of possible pneumonia. Radiology: Artificial Intelligence.
Bustos A et al. (2020). PadChest: a large chest x-ray image dataset with multi-label annotated reports. Medical Image Analysis.
Cohen JP et al. (2020). On the limits of cross-domain generalization in automated X-ray prediction. MIDL.
Erenstein H et al. (2025). The effect of simulated dose reduction on the performance of artificial intelligence in chest radiography. Journal of Imaging.
Autoria

Aline Lima — Análise e Desenvolvimento de Sistemas, iniciação científica.
Veronika - 