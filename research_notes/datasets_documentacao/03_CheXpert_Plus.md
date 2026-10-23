# DIAMANTE — CheXpert Plus

Data: 08/10/2026. Versão documental revisada. Pertence à mesma família do CheXpert original.

## Identificação e conteúdo

| Campo | Informação documentada |
|---|---|
| Origem | Extensão da coleção Stanford/CheXpert, apresentada em 2024. [P2] |
| Dimensão no portal atual | 223.462 pares imagem–laudo; 187.711 estudos; 64.725 pacientes. [P1] |
| Formato | DICOM; artigo também descreve PNG. [P1, P2] |
| Texto | 187.711 laudos, organizados em até 11 seções. [P1] |
| Metadados | Até 47 elementos DICOM e oito elementos de informação do paciente. [P1] |

**Divergência de versão:** o artigo v1 informa 223.228 imagens, enquanto o portal atual informa 223.462 pares. Registrar ambas as afirmações, sem presumir a causa. O inventário baixado determinará a contagem usada no trabalho. [P1, P2]

## População, identidade e referência

A família é predominantemente adulta, conforme documentação do original; a distribuição exata do Plus deve ser conferida separadamente. [C2] O vínculo paciente–estudo–imagem permite estruturar a auditoria, mas a correspondência de IDs entre versões precisa ser verificada.

Há rótulos derivados de laudos por CheXpert e CheXbert. Não representam releitura manual universal das imagens. A versão do extrator e a seção textual utilizada importam. [P2]

## Laudos e sintomas

O artigo informa `Clinical History` em 116.231 estudos e diferencia histórico, achados e impressão diagnóstica. Um laudo pode reunir informação de várias imagens do mesmo exame. [P2]

Proposta para o projeto: tratar o estudo com sintomas como frente complementar. Primeiro verificar se o histórico contém informação clínica disponível antes da interpretação, sua completude e eventuais menções ao diagnóstico. Usar a conclusão que gerou o rótulo como entrada para prever esse mesmo rótulo pode produzir vazamento. A existência de texto não garante uma coleção de sintomas completos, padronizados ou independentes.

## Aquisição e qualidade

O apêndice B lista, entre outros, `Manufacturer`, `ViewPosition`, `KVP`, `Exposure`, `ExposureIndex`, `WindowCenter` e `WindowWidth`. [P2]

Esses campos tornam a família prioritária para verificar aquisição e apresentação. Sua presença no esquema não garante preenchimento, unidade uniforme ou comparabilidade entre equipamentos. Tampouco DICOM significa necessariamente sinal bruto do detector.

Hipótese de trabalho a investigar: diferenças de apresentação podem estar associadas aos erros. Ainda não existe resultado do DIAMANTE demonstrando esse efeito, nem evidência de que exposição explique sozinha o brilho observado.

## Acesso e decisão

[Portal oficial Stanford AIMI](https://aimi.stanford.edu/datasets/chexpert-plus), com encaminhamento para [Redivis](https://stanford.redivis.com/datasets/5yyj-1a9f6ap0x?v=next). A página consultada não explicita todos os termos nos campos de licença; verificar as condições no fluxo oficial antes da obtenção. Acesso ainda não testado.

**Priorizar dentro da família Stanford**, por reunir imagem, texto e metadados. Não usar como teste externo de um modelo desenvolvido no CheXpert original sem demonstrar independência dos pacientes; mesmo pacientes separados continuam pertencendo à mesma origem institucional.

## Pendências para a fase 3

- Fixar a versão e reconciliar as contagens.
- Confirmar idade, projeção e vínculo entre arquivos.
- Medir disponibilidade dos campos técnicos, sem interpretar ausências como valores zero.
- Distinguir cada extrator de rótulos e seus estados de incerteza.
- Auditar interseções com modelos pré-treinados e com a versão original.

## Fontes

- **P1.** [Stanford AIMI — CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus).
- **P2.** Chambon et al. (2024). [CheXpert Plus — artigo](https://arxiv.org/abs/2405.19538); [v1 consultada](https://arxiv.org/html/2405.19538v1), seções 3–4 e apêndice B.
- **C2.** [Datasheet do CheXpert original](https://arxiv.org/abs/2105.03020). Não transferir automaticamente suas contagens para o Plus.
