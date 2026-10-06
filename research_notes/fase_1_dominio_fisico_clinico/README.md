# Fase 1 — Domínio físico, técnico e clínico da radiografia de tórax

Esta pasta documenta a **Fase 1** do projeto de auditoria de robustez e generalização de modelos de Deep Learning para pneumonia em radiografias de tórax.

## Objetivo da fase

Antes de treinar ou auditar modelos, compreender como a radiografia é formada, quais fatores alteram sua aparência e quais dessas variações representam:

- informação anatômica ou patológica;
- aquisição física;
- geometria;
- processamento digital;
- qualidade técnica;
- artefatos ou características do dataset.

O objetivo central foi evitar uma interpretação simplista do tipo:

> “mudança de brilho = mudança de exposição”.

A imagem final é consequência de uma cadeia muito maior:

```text
emissão de raios X
        ↓
interação com o paciente
        ↓
geometria da aquisição
        ↓
detector
        ↓
sinal digital
        ↓
pós-processamento
        ↓
imagem apresentada
        ↓
conversão / dataset
        ↓
pré-processamento para ML
        ↓
modelo
```

## Arquivos

- `1A_formacao_da_radiografia.md` — formação da imagem e origem das características visuais.
- `1B_aquisicao_exposicao_detector.md` — kVp, mAs, exposição, ruído, SNR/CNR, detector, DQE/MTF.
- `1C_geometria_aquisicao.md` — PA/AP, posicionamento, rotação, inspiração, SID e magnificação.
- `1D_pos_processamento_imagem_digital.md` — For Processing/For Presentation, LUT, windowing, DICOM, PNG/JPEG e intensidade.
- `1E_qualidade_interpretacao_clinica.md` — qualidade técnica, artefatos, cobertura anatômica, interpretabilidade e erros.
- `1F_sintese_matriz_metodologica.md` — consolidação científica da Fase 1 e matriz de decisão para as próximas fases.

## Conclusão geral

A principal conclusão metodológica é:

> **não vamos transformar todos os fatores identificados em augmentations.**

Alguns fatores serão avaliados por **perturbações sintéticas controladas**; outros serão tratados como **subgrupos reais**, **metadados**, **variáveis de controle** ou **limitações que não podem ser reconstruídas retrospectivamente**.

Essa distinção será usada para definir os datasets, o protocolo experimental e a auditoria dos modelos nas próximas fases.
## REFERÊNCIAS


## Referências

1. SCHAEFER-PROKOP, Cornelia; NEITZEL, Ulrich; VENEMA, Henk W.; UFFMANN, Martin; PROKOP, Mathias. Digital chest radiography: an update on modern technology, dose containment and control of image quality. *European Radiology*, v. 18, n. 9, p. 1818–1830, 2008. DOI: 10.1007/s00330-008-0948-3. PMID: 18431577. PMCID: PMC2516181.

2. ANDRIOLE, Katherine P.; RUCKDESCHEL, Thomas G.; FLYNN, Michael J.; HANGIANDREOU, Nicholas J.; JONES, A. Kyle; KRUPINSKI, Elizabeth; SEIBERT, J. Anthony; SHEPARD, S. Jeff; WALZ-FLANNIGAN, Alisa; MIAN, Tariq A.; POLLACK, Matthew S. ACR-AAPM-SIIM practice guideline for digital radiography. *Journal of Digital Imaging*, v. 26, n. 1, p. 26–37, 2013. DOI: 10.1007/s10278-012-9523-1. PMID: 22992867. PMCID: PMC3553368.

3. SHEPARD, S. Jeff; WANG, Jihong; FLYNN, Michael; GINGOLD, Eric; GOLDMAN, Lee; KRUGH, Kerry; LEONG, David L.; MAH, Eugene; OGDEN, Kent; PECK, Donald; SAMEI, Ehsan; WILLIS, Charles E. An exposure indicator for digital radiography: AAPM Task Group 116 (Executive Summary). *Medical Physics*, v. 36, n. 7, p. 2898–2914, 2009. DOI: 10.1118/1.3121505. PMID: 19673189. PMCID: PMC3908678.

4. DAVE, Jaydev K.; JONES, A. Kyle; FISHER, Ryan; HULME, Katie; RILL, Lynn; ZAMORA, David; WOODWARD, Andrew; BRADY, Samuel; MACDOUGALL, Robert D.; GOLDMAN, Lee; LANG, Susan; PECK, Donald; APGAR, Bruce; SHEPARD, S. Jeff; UZENOFF, Robert; WILLIS, Charles. Current state of practice regarding digital radiography exposure indicators and deviation indices: Report of AAPM Imaging Physics Committee Task Group 232. *Medical Physics*, v. 45, n. 11, p. e1146–e1160, 2018. DOI: 10.1002/mp.13212. PMID: 30255505.

5. DASEGOWDA, Giridhar; KALRA, Mannudeep K.; ABI-GHANEM, Alain S.; ARRU, Chiara D.; BERNARDO, Monica; SABA, Luca; SEGOTA, Doris; TABRIZI, Zhale; VISWAMITRA, Sanjaya; KAVIANI, Parisa; KAROUT, Lina; DREYER, Keith J. Suboptimal Chest Radiography and Artificial Intelligence: The Problem and the Solution. *Diagnostics*, v. 13, n. 3, art. 412, 2023. DOI: 10.3390/diagnostics13030412. PMID: 36766516. PMCID: PMC9914850.

6. ZECH, John R.; BADGELEY, Marcus A.; LIU, Manway; COSTA, Anthony B.; TITANO, Joseph J.; OERMANN, Eric Karl. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. *PLOS Medicine*, v. 15, n. 11, e1002683, 2018. DOI: 10.1371/journal.pmed.1002683. PMID: 30399157. PMCID: PMC6219764.

7. DAPAMEDE, Theo; LI, Frank; KHOSRAVI, Bardia; PURKAYASTHA, Saptarshi; TRIVEDI, Hari; GICHOYA, Judy. DICOM LUT is a Key Step in Medical Image Preprocessing Towards AI Generalizability. *Journal of Imaging Informatics in Medicine*, v. 38, n. 5, p. 3040–3048, 2025. DOI: 10.1007/s10278-025-01418-5. PMID: 39890738. PMCID: PMC12572459.

8. NATIONAL ELECTRICAL MANUFACTURERS ASSOCIATION — NEMA. *Digital Imaging and Communications in Medicine (DICOM) Standard — PS3.3: Information Object Definitions. Section C.7.6.3 — Image Pixel Module*. Rosslyn, VA: NEMA. Disponível em: DICOM Standard, seção C.7.6.3. Acesso em: 6 out. 2026.

9. NATIONAL ELECTRICAL MANUFACTURERS ASSOCIATION — NEMA. *Digital Imaging and Communications in Medicine (DICOM) Standard — PS3.3: Information Object Definitions. Section C.11.2 — VOI LUT Module*. Rosslyn, VA: NEMA. Disponível em: DICOM Standard, seção C.11.2. Acesso em: 6 out. 2026.

10. AMERICAN ASSOCIATION OF PHYSICISTS IN MEDICINE — AAPM. *An Exposure Indicator for Digital Radiography: Report of AAPM Task Group 116*. AAPM Report No. 116. College Park, MD: American Association of Physicists in Medicine, 2009. DOI: 10.37206/111.

11. AMERICAN ASSOCIATION OF PHYSICISTS IN MEDICINE — AAPM. *Current State of Practice Regarding Digital Radiography Exposure Indicators and Deviation Indices: Report of AAPM Imaging Physics Committee Task Group 232*. AAPM Report No. 232. Alexandria, VA: American Association of Physicists in Medicine, 2018. DOI: 10.1002/mp.13212.