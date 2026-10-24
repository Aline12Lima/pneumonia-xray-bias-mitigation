# Auditoria PadChest_01

Data: 2026-10-17 · Fase: F3 (passos 3.1a e 3.7) · Responsável: Aline Lima
Escopo: sem imagens. Usou só o CSV de rótulos, a planilha de contagens e as 52 listagens dos zips.
Decisão gerada: D-025.

**Resultado:** o PadChest entra como teste externo pelos pacotes 0–11 e 54 (282 GB, 796 positivos), que formam o subconjunto rotulado por médicos, concentrado em 2014–2017.

## Arquivos

| Item | Caminho |
| --- | --- |
| Script | `src/data/padchest_audit.py` |
| Saída completa do script | `docs/dataset_notes/auditoria_padchest_01_saida.md` |
| Tabelas de apoio | `results/tables/padchest/` (`per_zip_summary.csv`, `dicom_fields_completeness.csv`, `candidate_vs_rest.csv`, `candidate_physical_fields.csv`) |
| Entradas | `data/raw/padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz`, `Verify_Zips_ImageCounts.csv.xlsx`, `unzip-l/*.txt` |

## Integridade

- 160.861 imagens no CSV, nas listagens e na planilha de contagens. As três fontes batem.
- `ImageDir` igual ao pacote da listagem em 100% das imagens.
- 52 pacotes (0–50 e 54), 1.058 GB no total, cerca de 6,6 MB por imagem. O pacote 50 tem 1 imagem.

## Recorte provisório

Critério provisório para "uma frontal por estudo": PA > AP > AP_horizontal, depois ImageID. A regra final é decisão do passo 4.3.

| Etapa | n |
| --- | ---: |
| Todas as imagens | 160.861 |
| Frontais (PA, AP, AP_horizontal) | 110.633 |
| Sem marca `Pediatric=PED` | 110.383 |
| Idade ≥ 18 (ano do estudo − ano de nascimento) | 105.108 |
| Com rótulo | 104.585 |
| Uma frontal por estudo | 103.036 |

Resultado: 103.036 estudos, 62.714 pacientes, 4.079 positivos (4,0%).

Associações com o rótulo, que importam para o T1 e para os pesos do eixo C:

| Variável | Prevalência de pneumonia |
| --- | --- |
| PA / AP / AP_horizontal | 3,1% / 8,0% / 8,7% |
| Philips / Imaging Dynamics | 6,25% / 1,8% |
| Rótulo médico / RNN | 3,2% / 4,2% |
| Masculino / feminino | 4,6% / 3,4% |

## Os pacotes não são aleatórios

| Característica | V de Cramér com o pacote |
| --- | ---: |
| Método de rotulagem | 0,993 |
| Ano do estudo | 0,566 |
| Fabricante | 0,293 |
| Projeção | 0,118 |
| Faixa etária | 0,062 |
| Pneumonia | 0,044 |
| Sexo | 0,030 |

- Pacotes 0–11 e 54: 100% rotulados por médico, sobretudo 2015–2017.
- Pacote 12: misto (14% rotulado por médico).
- Pacotes 13–49: 100% RNN, cada um concentrado em poucos anos entre 2009 e 2015.
- 28,8% dos pacientes têm estudos em mais de um pacote.

Conclusão: sortear pacotes inteiros sorteia blocos de período e de método de rotulagem. Com sorteio, seriam precisos 10 pacotes na mediana (207 GB) para 750 positivos, com resultado dependente do sorteio.

## Subconjunto escolhido: pacotes 0–11 e 54

| Item | Candidato | Restante | Todos |
| --- | ---: | ---: | ---: |
| Estudos (sem `exclude`) | 24.775 | 77.369 | 102.144 |
| Pacientes | 22.988 | 46.803 | 62.373 |
| Positivos | 796 | 3.283 | 4.079 |
| Prevalência (%) | 3,2 | 4,2 | 4,0 |
| PA (%) | 91,7 | 81,5 | 84,0 |
| AP_horizontal (%) | 7,0 | 14,9 | 13,0 |
| Philips (%) | 74,0 | 40,5 | 48,6 |
| Feminino (%) | 52,3 | 49,1 | 49,9 |
| Idade média (anos) | 61,9 | 61,2 | 61,4 |
| Estudos de 2015–2017 (%) | 75,8 | 0,4 | 18,7 |
| Rótulo médico (%) | 100 | 0,4 | 24,5 |

- Download: 13 pacotes, 282,1 GB.
- Rótulo `exclude`: 57 estudos removidos, nenhum positivo.
- Pneumonia atípica: 16 estudos (812 positivos se contados).
- Portão G3: atingido, com 796 positivos (folga de 46).
- Ficam de fora 290 estudos com rótulo médico no pacote 12 (9 positivos, 22,5 GB): não compensa baixar.

Positivos no candidato por estrato:

| Estrato | Estudos | Positivos | Prevalência (%) |
| --- | ---: | ---: | ---: |
| PA | 22.714 | 671 | 2,95 |
| AP_horizontal | 1.742 | 98 | 5,63 |
| AP | 319 | 27 | 8,46 |
| Philips | 18.343 | 716 | 3,90 |
| Imaging Dynamics | 6.432 | 80 | 1,24 |
| 2014 | 5.869 | 178 | 3,03 |
| 2015 | 6.868 | 243 | 3,54 |
| 2016 | 6.777 | 207 | 3,05 |
| 2017 | 5.145 | 161 | 3,13 |

## Passo 3.7: campos DICOM no CSV

O CSV traz campos físicos, mas cada fabricante registra coisas diferentes. Valores no candidato, com zeros contados como ausentes:

| Campo | Imaging Dynamics (útil %) | Philips (útil %) |
| --- | ---: | ---: |
| Corrente (`XRayTubeCurrent_DICOM`) | 99,8 | 0 (só zeros ou vazio) |
| mAs (`ExposureInuAs_DICOM`) | 99,7 | 44,9 |
| Tempo (`ExposureTime`) | 99,7 | 44,9 |
| Índice de exposição relativa | 51,5 (mediana −1,2) | 99,3 (mediana 353) |
| Janela, centro / largura (medianas) | 2189 / 2363 | 2047 / 4095 |

- Não há kVp em nenhum fabricante.
- As imagens são PNG de 16 bits, com `BitsStored` de 12 ou 10.
- O índice de exposição tem escalas diferentes nos dois fabricantes. Confirma a D-013: análise física só dentro do fabricante.
- No Philips, a janela cobre a faixa inteira de 12 bits, então converter "pela janela" equivale a uma escala linear. Reforça a comparação de duas conversões do passo 3.12.
- Eixo A físico no PadChest: no Philips, com o índice de exposição relativa (principal) e o mAs (secundário, 45%); no Imaging Dynamics, só exploratório, por ter 80 positivos.

## Pendências

- [ ] Conferir no artigo do PadChest como foi escolhida a amostra rotulada por médicos
- [ ] Baixar os pacotes 0–11 e 54 e extrair só as imagens do recorte (cerca de 25 mil)
- [ ] Recontar positivos após a inspeção das imagens; se cair abaixo de 750, aplicar a regra de 500 com 2 faixas do G3
- [ ] Fechar a regra de "primeira frontal" (passo 4.3)
