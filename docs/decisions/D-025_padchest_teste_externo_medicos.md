# D-025 — Teste externo do PadChest: subconjunto rotulado por médicos

Data: 2026-10-17 · Fase: F3 · Estado: Tomada

Contexto: a auditoria PadChest_01 mostrou que os 52 pacotes do PadChest não são aleatórios. Eles são blocos de método de rotulagem (V de Cramér = 0,99) e de ano (V = 0,57). Sortear pacientes exigiria baixar cerca de 1 TB.

Opções consideradas:
- A. Pacotes 0–11 e 54, o subconjunto rotulado por médicos (282 GB, 796 positivos)
- B. Sortear pacotes inteiros (mediana de 10 pacotes, 207 GB, cerca de 750 positivos)
- C. Sortear pacientes em todo o PadChest (cerca de 1 TB)

Decisão: opção A.
- Teste externo = pacotes 0–11 e 54, frontais de adultos, uma por estudo, sem rótulo `exclude`.
- Positivo = rótulo `pneumonia`.
- Pneumonia atípica (16 estudos): fora da análise primária; positiva na análise de sensibilidade.
- O pacote 12 não é baixado (290 estudos com rótulo médico, 9 positivos, 22,5 GB).

Por quê: rótulo de referência feito por médicos; 796 positivos, o que atinge o G3; critério fixado antes de ver qualquer imagem. Evidência em `docs/dataset_notes/auditoria_padchest_01.md`.

Consequência:
- Substitui a D-006 ("recorte aleatório de pacientes").
- O teste passa a ser descrito como "PadChest 2014–2017, rotulado por médicos". Ele difere do resto do PadChest: mais PA (91,7% contra 81,5%), mais Philips (74% contra 40%) e prevalência menor (3,2% contra 4,2%). Isso entra nas limitações.
- O controle do passo 7.7 (subconjunto médico do PadChest) fica embutido no próprio teste.
- Eixo A físico no PadChest: só dentro do fabricante; no Imaging Dynamics, exploratório (80 positivos).
- Se a inspeção das imagens deixar menos de 750 positivos, aplica-se a regra de 500 com 2 faixas do G3.

Reversível até: G4
