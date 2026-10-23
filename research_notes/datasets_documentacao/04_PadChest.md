# DIAMANTE — PadChest

Data: 08/10/2026. Selecionada para auditoria; distribuição completa ainda não obtida.

## Origem, população e conteúdo

| Campo | Informação documentada |
|---|---|
| Origem | Hospital Universitario de San Juan, Alicante, Espanha; 2009–2017. [D1] |
| Conjunto final | 160.868 imagens, 109.931 estudos e 67.625 pacientes na tabela 7 do preprint. [D2] |
| Idade | 0–105 anos; média 58,5 e mediana 62. [D2] |
| Imagens | Tórax em PNG, acompanhado de informações extraídas de DICOM. [D2] |
| Projeções | PA, lateral, AP vertical/horizontal e categorias especiais. [D2] |
| Vocabulário no portal | 174 achados, 19 diagnósticos diferenciais e 104 localizações. [D1] |

Há inconsistências numéricas no preprint, incluindo contagens de pacientes em diferentes seções. A tabela do conjunto final é nossa referência documental; o pacote real deverá confirmá-la. [D2]

## Rotulação e laudos

Segundo o portal, aproximadamente 27% dos laudos foram anotados manualmente por médicos e os demais por um modelo supervisionado. [D1] Isso é anotação de texto clínico, não releitura visual independente de todas as imagens.

O campo `Report` é descrito como trecho da interpretação radiográfica, processado, com tokenização e redução de palavras. `MethodLabel` distingue procedência manual/automática. [D2]

Para o DIAMANTE, comparar essas procedências é uma possibilidade, não uma garantia de grupos equivalentes: época e composição clínica também podem diferir. Laudo não deve ser tratado como lista completa de tudo que a imagem contém.

## Pediatria, identidade e qualidade

`PatientID`, `StudyID` e `PatientBirth` permitem investigar agrupamento e idade aproximada. `Pediatric` identifica um protocolo de aquisição, descrito para até três anos; não enumera todos os menores de idade. [D2]

O portal lista fabricante, modalidade, exposição, corrente e tempo de exposição entre os metadados. [D1] O artigo descreve processamento prévio com janela DICOM e inclui estudos de técnica subótima. [D2]

Consequência metodológica: investigar o conteúdo antes de afirmar que a distribuição contém imagens “originais sem tratamento”. A idade derivada apenas de ano de nascimento não distingue meses de vida. Um campo técnico ou anotação de qualidade ausente não equivale a exame normal ou tecnicamente adequado.

## Acesso

[BIMCV — página oficial PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/). O recurso completo exige solicitação formal e acordo de uso; o portal anuncia cerca de 1 TB. Há uma amostra indicada no Kaggle, que não substitui a coleção completa. [D1]

O acesso não foi solicitado nesta revisão. Registrar autorização, versão, arquivos disponíveis e condições de armazenamento antes de iniciar a fase prática.

## Decisão e pendências

**Manter**, especialmente para investigar metadados e outra origem hospitalar. Considerar o subconjunto pediátrico apenas após medir sua composição.

- Conferir a contagem por paciente e a diferença entre protocolo pediátrico e idade.
- Verificar preenchimento e unidades dos campos de aquisição.
- Identificar quais projeções foram inferidas e quais têm informação de origem.
- Separar achado radiográfico, diagnóstico diferencial e referência de normalidade.
- Definir negativos sem equiparar ausência de rótulo a ausência clínica.
- Conferir o esquema exato de labels e qualidade da versão recebida.

## Fontes

- **D1.** [BIMCV — PadChest, descrição, acesso e campos](https://bimcv.cipf.es/bimcv-projects/padchest/).
- **D2.** Bustos et al. [PadChest: A large chest x-ray image dataset with multi-label annotated reports](https://arxiv.org/abs/1901.07441). Preprint de 2019; artigo publicado em 2020, [DOI](https://doi.org/10.1016/j.media.2020.101797). Referências úteis: tabelas 5–9 e método de preparação dos dados.

Literatura de uso compartilhado com CheXpert: [Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning, 2022](https://doi.org/10.1038/s41551-022-00936-9). Ver ficha bibliográfica 08.
