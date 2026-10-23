# DIAMANTE — PediCXR / VinDr-PCXR

Data: 08/10/2026. **Base pediátrica principal selecionada para auditoria.** Os dois nomes identificam o mesmo recurso neste planejamento.

## Origem, população e formato

| Campo | Informação documentada |
|---|---|
| Origem | Vietnã, 2020–2021; método identifica o Phu Tho Obstetric & Pediatric Hospital e extração de PACS. [V1] |
| Dimensão | 9.125 imagens/estudos; artigo informa um estudo por paciente. [V1] |
| População | Descrita como menores de dez anos; medianas 1,71 ano no treino e 1,69 no teste. [V1] |
| Formato/projeção | DICOM; artigo descreve PA. [V1] |
| Metadados | Idade, sexo e atributos de imagem entre os campos preservados. [V1] |
| Referência | Anotação visual especializada: 36 achados e 15 doenças, com localização de achados. [V2] |
| Organização | 7.728 imagens no treino e 1.397 no teste; CSVs para caixas e rótulos por imagem. [V2] |

O artigo informa ausência de informação clínica associada e exclusão de imagens de baixa qualidade. Não descreve uma distribuição de laudos completos. [V1]

## Inconsistências que não devem ser escondidas

O texto original alterna referências a um e a múltiplos hospitais. O portal também diverge sobre número de anotadores e contém trecho que chama a população de adulta, contrariando a própria descrição pediátrica. [V1, V2]

Um estudo de uso de 2024 descreve seu subconjunto PediCXR como 0–12 anos, divergindo do limite do artigo de apresentação. [V3] Portanto, “menores de dez” fica registrado como **declaração da fonte original**, não como máximo verificado dos arquivos.

## O que podemos investigar

Recomendação: priorizar esta base para verificar se há suporte a análises por idade infantil, apresentação e erro. A diversidade de rótulos permite estudar casos com outras alterações, se houver amostra suficiente.

A seleção de imagens de melhor qualidade limita uma conclusão sobre toda a prática pediátrica. Desempenho nesta coleção não prova robustez em radiografias rejeitadas ou em outras populações. Testes com alterações artificiais de imagem, se posteriormente aprovados, não substituirão evidência de qualidade clínica real.

Também não devemos presumir que toda configuração de exposição/detector foi preservada. Verificar a lista de tags e seu preenchimento antes de prometer uma análise física equivalente à possível nas bases com metadados mais amplos.

## Identidade e compatibilidade

A afirmação de um exame por paciente vem dos autores; o identificador público de imagem deriva do SOP Instance UID. [V1] Na auditoria, confrontar essa descrição com duplicações e estrutura dos arquivos. Não tentar recuperar identidade civil.

Para uma tarefa de pneumonia, conferir os nomes e a hierarquia de diagnósticos no CSV. Não agrupar automaticamente bronquite, consolidação, opacidade e pneumonia. A referência infantil deve ser comparada à definição dos outros datasets antes de uma avaliação conjunta.

## Acesso

[PhysioNet — VinDr-PCXR v1.0.0](https://physionet.org/content/vindr-pcxr/1.0.0/). A página atual exige usuário registrado e assinatura do acordo de uso; informa licença restrita de dados de saúde. [V2] Não houve cadastro nem download nesta etapa.

## Pendências para a fase 3

- Confirmar intervalo etário real e unidade da idade; investigar a divergência 10/12 anos.
- Conferir se a projeção descrita como PA corresponde aos arquivos e à documentação disponível.
- Confirmar um paciente por estudo e registrar limitações de verificação independente.
- Contar os alvos relevantes por idade e verificar classes raras.
- Conciliar versões e inconsistências do portal; consultar autores se a divergência afetar a análise.

## Fontes

- **V1.** Pham et al. (2023). [PediCXR: An open, large-scale chest radiograph dataset for interpretation of common thoracic diseases in children](https://doi.org/10.1038/s41597-023-02102-5); [texto dos autores](https://arxiv.org/pdf/2203.10612), tabelas 2–3 e “Data Records”.
- **V2.** [PhysioNet — descrição, arquivos e licença](https://physionet.org/content/vindr-pcxr/1.0.0/).
- **V3.** [Self-Supervised Learning for Building Robust Pediatric Chest X-ray Classification Models (2024)](https://arxiv.org/html/2409.00231v1), tabela 1 e apêndice A.1. Estudo de aplicação; não resolve a divergência com a fonte de origem.
