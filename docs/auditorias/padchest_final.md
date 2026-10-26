# Auditoria final — PadChest

**Fase:** F3 — Datasets e auditoria
**Base:** PadChest (Hospital Universitario de San Juan, Alicante), pacotes 0–11 e 54
**Execução:** 22 e 23/10/2026, Redivis, workflow `auditoria_padchest`
**Auditoria anterior:** `padchest_preliminar.md` (17/10, só metadados), que definiu o recorte
**Avaliadora da máscara:** Aline Lima

**Estado: base aprovada.** As quatro exigências para uma base entrar no estudo estão atendidas: positivos suficientes no teste, ID de paciente utilizável, máscara do pulmão aceitável em pelo menos 95 de 100 imagens, e termos de uso compatíveis.

---

## 1. Resumo

- **Papel:** só teste externo, para os três modelos. Nenhum modelo treina no PadChest.
- **Base final:** 24.770 estudos (uma imagem frontal por estudo), cerca de 22.987 pacientes, **796 positivos (3,21%)**. Atinge a meta de 750, com folga de 46.
- **Todas as 24.773 imagens do recorte abriram**, e as dimensões batem 100% com o CSV. Nenhuma duplicata exata.
- **Os PNG estão reescalados para 16 bits:** a profundidade original (10 ou 12 bits) se mantém, mas a escala foi esticada. A janela do CSV só funciona depois dessa conversão.
- **Fabricante, saturação e prevalência andam juntos:** o Imaging Dynamics satura o branco em 59% das imagens e tem 1,24% de pneumonia; o Philips, 0% e 3,90%.
- **Máscara do pulmão:** 97 de 100 aceitáveis.
- **Termos de uso estritos:** dados sempre privados; só resultados agregados podem ser publicados.

---

## 2. Termos de uso

O acordo de uso do PadChest (`LICENSE.md` da distribuição do BIMCV) é **compatível com o estudo, com quatro condições**.

| Cláusula | O que diz | O que significa para o estudo |
|---|---|---|
| 1 | Uso gratuito só para pesquisa; uso não acadêmico precisa de aprovação | Iniciação científica é pesquisa acadêmica: permitido |
| 4 | Pode fazer uma cópia para uso pessoal de pesquisa | Dados num notebook ou dataset privado: permitido |
| 5 | Proibido distribuir, publicar ou reproduzir qualquer parte dos dados sem permissão por escrito | Dataset e workflow sempre restritos à autora; nenhum dado por imagem é publicado |
| 6 | Proibido compartilhar o link de download fora da organização | O link não vai para documentos, repositório nem workflow compartilhado |
| 7 | Proibido criar obras derivadas, exceto modelos e algoritmos | A cópia de 320 px e as medidas de qualidade ficam só internas, para avaliar o modelo |
| 2 | Sugere publicar código e pesos e citar o artigo | Citar Bustos et al. (2020) |
| 10 | Proibido tentar reidentificar pacientes | Nada no estudo faz isso |

**Consequência para o plano:** o passo de arquivamento com DOI (fase de reprodutibilidade) previa publicar as medidas de qualidade por imagem. No PadChest, só podem ser publicados resultados agregados (tabelas, métricas, figuras) e o código.

---

## 3. Recorte e fluxograma

O recorte foi definido na auditoria preliminar (17/10), antes de ver qualquer imagem: **pacotes 0–11 e 54**, que são 100% rotulados por médicos. Os pacotes do PadChest não são aleatórios (são blocos de método de rotulagem e de ano), e esse subconjunto tem o rótulo de referência mais forte.

| Etapa | Imagens | Estudos | Pacientes | Positivos (por estudo) |
|---|---|---|---|---|
| Todo o PadChest | 160.861 | 109.931 | 67.625 | 5.037 |
| Pacotes 0–11 e 54 | 38.606 | 26.112 | 24.217 | 1.054 |
| Frontais (PA, AP, AP deitado) | 26.126 | 25.786 | 23.965 | 1.026 |
| Idade de 18 a 100 anos | 25.129 | 24.830 | 23.037 | 796 |
| Sem rótulo "exclude" | 25.072 | 24.773 | 22.987 | 796 |
| Uma frontal por estudo e sexo registrado | **24.770** | **24.770** | **cerca de 22.987** | **796** |

- **Uma frontal por estudo:** PA, depois AP, depois AP deitado.
- **Sexo "O":** 3 imagens, nenhuma positiva, excluídas pela mesma regra do Plus.
- **Pneumonia atípica:** 16 estudos a mais; positivos na análise de sensibilidade (812 no total).
- **O teste é "PadChest 2014–2017, rotulado por médicos".** Ele difere do resto do PadChest: mais PA (91,7% contra 81,5%), mais Philips (74% contra 40%) e prevalência menor (3,2% contra 4,2%). Isso entra nas limitações.

---

## 4. Download e integridade

- **Processamento em fluxo:** os 13 pacotes (282 GB) foram baixados um por vez dentro do notebook, conferidos, convertidos e apagados. Nenhuma imagem original ficou guardada.
- **As 24.773 imagens do recorte abriram, todas.**
- **Formato:** PNG de 16 bits, um canal, MONOCHROME2, sem compressão com perda.
- **Dimensões:** batem 100% com o CSV. Mediana de 2000 × 2000 no Imaging Dynamics e de 2545 × 2706 no Philips.
- **Duplicatas exatas:** nenhuma.

---

## 5. Escala dos pixels

**O CSV registra 10 ou 12 bits, mas os PNG ocupam toda a escala de 16 bits.**

- **Valores diferentes por imagem:** cerca de 2.300 a 3.700 nas imagens de 12 bits e cerca de 1.000 nas de 10 bits. A profundidade original se mantém; só a escala foi esticada.
- **Valor máximo:** 65.519 nas de 12 bits e 65.471 nas de 10 bits. Bate exatamente com uma reescala linear: 12 bits multiplicados por 16,004; 10 bits por cerca de 64,06.
- **Janela do CSV:** na escala original cobria de 0% a 8% dos pixels; convertida para a escala dos PNG, cobre **100% da imagem no Philips e 69% no Imaging Dynamics**.

**Conclusão:** a correspondência entre o CSV e os PNG é conhecida. A janela pode ser usada, desde que convertida.

**Conversão para 8 bits:** a mesma regra das três bases, mínimo-máximo por percentis 0,5–99,5. A janela do CSV reescalada entra como a segunda conversão, em análise de sensibilidade.

---

## 6. Fabricante e saturação

| | Imaging Dynamics | Philips |
|---|---|---|
| Imagens | 6.433 | 18.340 |
| Pixels no valor máximo (mediana) | **1,84%** | 0,01% |
| Imagens com mais de 1% dos pixels no valor máximo | **59%** | 0% |
| Prevalência de pneumonia | **1,24%** | **3,90%** |

**O Imaging Dynamics satura o branco em mais da metade das imagens e tem um terço da prevalência do Philips.** Qualidade da imagem, fabricante e rótulo andam juntos. Por isso as análises de qualidade são feitas sempre dentro do fabricante, e o teste de atalhos (T1) inclui o fabricante.

### Parâmetros físicos (do CSV, zeros contados como ausentes)

| Campo | Imaging Dynamics (útil) | Philips (útil) |
|---|---|---|
| Corrente | 99,8% | 0% |
| Produto corrente-tempo e tempo | 99,7% | 44,9% |
| Índice de exposição relativa | 51,5% (mediana −1,2) | 99,3% (mediana 353) |
| kVp | ausente | ausente |

O índice de exposição tem escalas diferentes nos dois fabricantes, então a análise física fica dentro de cada fabricante. No Imaging Dynamics ela é só exploratória (80 positivos).

---

## 7. Rótulos

- **Positivo:** rótulo "pneumonia" no laudo, atribuído por médico (100% do recorte).
- **Negativo:** ausência do rótulo "pneumonia". Não significa exame normal.
- **Sensibilidade:** incluir "atypical pneumonia".
- **O rótulo é por estudo**, e cada estudo contribui com uma única imagem frontal.

---

## 8. Composição da base final

| Subgrupo | Imagens | Prevalência |
|---|---|---|
| PA | 22.713 | 2,95% |
| AP (AP e AP deitado) | 2.060 | 6,1% (AP 8,5%; AP deitado 5,6%) |
| Feminino | 12.956 | 2,86% |
| Masculino | 11.814 | 3,60% |
| 18–39 anos | 2.024 | 5,39% |
| 40–59 anos | 8.487 | 2,45% |
| 60–74 anos | 8.371 | 3,07% |
| 75+ anos | 5.891 | 3,77% |

- **A projeção segue a direção do RSNA** (mais pneumonia no AP), com intensidade bem menor. O CheXpert Plus é a única base invertida.
- **A idade repete o padrão em U** visto no Plus.
- **Homens têm um pouco mais** de pneumonia que mulheres.

---

## 9. Máscara do pulmão

### 9.1 Modelo e procedimento

O mesmo das outras duas bases: `lungs-segmentation` resnet34, treinado só em Montgomery e Shenzhen; entrada por percentis 0,5–99,5, 512 × 512, limiar 0,2, maior componente de cada pulmão, união dos dois. O CheXmask não pode ser usado aqui, porque foi treinado com parte do PadChest.

### 9.2 Inspeção

- **Amostra:** 50 AP (AP e AP deitado) e 50 PA, semente 2026, já guardadas na primeira passada (sem novo download).
- **Às cegas:** as folhas mostravam só o número e a projeção, sem o rótulo.
- **Critério fixado antes de olhar:** o mesmo das outras bases.

| Caso | Veredito | Motivo |
|---|---|---|
| #5 | Falha | O pulmão mais comprometido (cerca de 80%, muito opaco) ficou reduzido a uma ilha |
| #15 | Falha | A máscara caiu no centro do tórax; os dois pulmões estão muito opacos |
| #40 | Falha | No pulmão direito, pegou só a parte escura e deixou de fora a parte clara comprometida |
| #8, #10, #93 | Aceitável | Corretas quando ampliadas |
| #43 | Aceitável (revisto em 25/10) | O alerta automático marcou um pulmão bem menor que o outro, mas a própria imagem mostra só um lado do tórax (recorte ou colimação na aquisição); a máscara segue a anatomia visível |

**Resultado: 97 de 100 aceitáveis.** Passa no critério de 95.

**Observação sobre o #43:** imagens recortadas que mostram só um lado do tórax existem na base. A máscara não é o problema nesses casos; o enquadramento da imagem é. A medida `q_enquadramento` e a razão entre os pulmões ajudam a identificá-las.

**Padrão das falhas:** imagens AP com pulmões muito opacos, onde o modelo marca o pulmão com ar e perde o pulmão doente. É o mesmo padrão do Plus e do RSNA. Nenhuma imagem da inspeção estava invertida.

---

## 10. Espaço no Redivis

| Item | Tamanho |
|---|---|
| Cópia de 320 px de todo o recorte | 4,11 GB |
| Amostra de inspeção | 0,27 GB |

O modelo base não precisa baixar os 282 GB de novo para treinar e avaliar. Só as medidas de qualidade na resolução original exigiriam um novo download (seção 12).

---

## 11. Situação das exigências para a base entrar no estudo

| Exigência | Situação |
|---|---|
| Positivos suficientes no teste (750 com 3 faixas) | **Atende:** 796 |
| ID de paciente utilizável | **Atende:** `PatientID` do CSV |
| Máscara aceitável em pelo menos 95 de 100 | **Atende:** 96/100 (97 se o #43 for aceitável) |
| Termos de uso compatíveis | **Atende, com condições:** dados privados, só resultados agregados publicados, link não compartilhado |

---

## 12. Pendências que não impedem a aprovação

| Item | Onde será feito |
|---|---|
| Medidas de qualidade na resolução original | Exige baixar os 282 GB de novo (cerca de 4 h) ou registrar a medida em resolução reduzida como decisão |
| Duplicatas por pixel (hash perceptual) | Na mesma passada das medidas |
| Comparar a área da máscara com o rótulo | Teste geral da máscara (500 positivos e 500 negativos do RSNA) e passada completa |
| Confirmar o veredito do #43 | Aline |

---

## 13. Decisões

| Decisão | Conteúdo |
|---|---|
| **F3 › PadChest › Termos** | Dados sempre privados; publicar só resultados agregados e o código; link de download não compartilhado |
| **F3 › PadChest › Recorte** | Pacotes 0–11 e 54 (100% rotulados por médicos) como teste externo |
| **F3 › PadChest › Exclusões** | Fora dos pacotes escolhidos; idade fora de 18 a 100; projeções não frontais; rótulo "exclude"; sexo "O" |
| **F3 › PadChest › Escala** | PNG reescalados para 16 bits; a janela do CSV vale só depois da conversão |
| **F3 › PadChest › Fabricante** | Análises de qualidade e física sempre dentro do fabricante |

---

## Referências

- Bustos A, Pertusa A, Salinas JM, de la Iglesia-Vayá M (2020). PadChest: a large chest x-ray image dataset with multi-label annotated reports. *Medical Image Analysis* 66, 101797. https://doi.org/10.1016/j.media.2020.101797
- BIMCV. PadChest: descrição e acesso. https://bimcv.cipf.es/bimcv-projects/padchest/
- PadChest Dataset Research Use Agreement (`LICENSE.md` da distribuição).
- Modelo de máscara: https://github.com/alimbekovKZ/lungs_segmentation e https://github.com/alimbekovKZ/lungs_segmentation_train
