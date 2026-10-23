# DIAMANTE — CheXpert original e datasheet

Data: 08/10/2026. Selecionado como referência da família Stanford; priorizar a extensão Plus para auditoria de aquisição e laudos.

## Identificação

| Campo | Informação documentada |
|---|---|
| Origem | Sistemas clínicos de Stanford Hospital, EUA; aquisições de 2002–2017. Não é uma coleção originalmente fotografada por smartphone. [C2] |
| Dimensão original | 224.316 radiografias de 65.240 pacientes. [C1] |
| Conteúdo | Tórax frontal e lateral; imagens distribuídas em JPEG. [C1, C2] |
| Referência | 14 observações; extração automática dos laudos no treinamento. [C1] |
| Avaliação original | Validação e teste anotados por radiologistas a partir das imagens. [C1] |

O datasheet é **documentação**, não outro dataset. A publicação original e o datasheet devem ser lidos em conjunto.

## População e pediatria

O datasheet descreve predomínio adulto. No treinamento, sua tabela VIII registra 3 imagens em 0–1 ano e 766 em 13–18 anos; a faixa adolescente inclui 18 anos. Idade é fornecida em anos completos. [C2]

Esses números não sustentam uma representação ampla da infância. Para o DIAMANTE, registros muito raros em idades inesperadas devem ser auditados antes de serem usados como evidência de cobertura pediátrica. Não vamos classificar automaticamente todos os exames como adultos nem assumir que esses poucos registros são suficientes.

## Rótulos e laudos

No treinamento, o significado de cada valor é: positivo `1`, negativo `0`, incerto `-1` ou sem menção, em branco. [C2] Pneumonia, consolidação e opacidade pulmonar são observações distintas. [C1, C2]

Proposta para auditoria: manter esses estados separados e contar sua distribuição. A ausência de menção não comprova ausência clínica. Um estudo que transforma indeterminados em positivos ou negativos precisa justificar essa decisão e avaliar sua influência.

Os laudos completos não integravam a distribuição original; a extensão Plus amplia o acesso a texto. [C2, P1] Uma etiqueta extraída de laudo também pode se referir ao exame com várias imagens, não a uma alteração identificável em qualquer imagem isolada.

## Metadados, aquisição e identificação

Há organização por paciente/estudo e informações de sexo, idade, frontal/lateral e AP/PA. O datasheet informa diversidade de equipamentos, sem identificação de equipamento por imagem na distribuição original. [C2]

Recomendação: não misturar JPEG reduzido, JPEG em outra resolução e DICOM Plus como se fossem a mesma apresentação. A comparação entre versões pode ser uma análise de apresentação, mas não validação externa entre hospitais.

## Acesso

- [Página oficial CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/).
- [Datasheet solicitado](https://arxiv.org/abs/2105.03020).

Registrar versão, pacote, condições de uso e disponibilidade real dos rótulos de avaliação no momento da obtenção. Esta revisão não executou cadastro ou download.

## Decisão e pendências

**Manter como documentação da família e referência comparativa.** Na fase 3, priorizar Plus, sem contar as duas versões como populações independentes.

Verificar: correspondência entre versões; agrupamento por paciente; idade; elegibilidade frontal; definição de negativos; procedência de cada conjunto de rótulos; interseção com dados de pré-treinamento. O papel A/B/C ainda não está atribuído.

## Fontes

- **C1.** Irvin et al. (2019). [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031).
- **C2.** Garbin et al. (2021). [Structured dataset documentation: a datasheet for CheXpert](https://arxiv.org/abs/2105.03020), especialmente tabelas IV e VIII e “Collection Process”.
- **P1.** [Stanford AIMI — CheXpert Plus](https://aimi.stanford.edu/datasets/chexpert-plus).
