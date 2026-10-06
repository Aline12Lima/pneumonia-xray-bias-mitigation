# 1F — Síntese da Fase 1 e matriz metodológica

## 1. Conclusão científica da Fase 1

A radiografia observada pelo modelo é resultado de múltiplas camadas:

```text
PACIENTE
│
├── anatomia
└── patologia
       ↓
INTERAÇÃO COM RAIOS X
       ↑
   kVp / mAs
       ↓
GEOMETRIA
PA/AP / posição / inspiração / rotação
       ↓
DETECTOR
DQE / MTF / ruído
       ↓
PÓS-PROCESSAMENTO
LUT / window / enhancement
       ↓
IMAGEM FINAL
       ↓
QUALIDADE
ruído / blur / cobertura / artefatos
       ↓
CONVERSÃO / DATASET
DICOM / PNG / compressão / labels
       ↓
PREPROCESSAMENTO ML
       ↓
MODELO
       ↓
previsão / confiança / erro
```

Portanto:

> **uma característica visual observada não identifica automaticamente sua causa física.**

---

# 2. Evolução da hipótese

## Formulação inicial

> Luminosidade influencia os erros do modelo.

## Formulação após a Fase 1

> Em que medida modelos de classificação de pneumonia permanecem robustos quando características não patológicas da imagem variam em decorrência de aquisição, geometria, processamento, qualidade e domínio?

Pergunta complementar:

> Quando o modelo falha, seus erros estão associados à evidência patológica, à degradação da qualidade ou a características técnicas e específicas do domínio?

A luminosidade/intensidade permanece relevante, porém como **uma variável observável entre várias**.

---

# 3. Conclusão metodológica central

> **Não vamos transformar todos os fatores identificados em augmentations.**

Cada fator será tratado conforme sua natureza física e a possibilidade real de reconstrução.

Alguns serão:

- **observados**;
- **medidos**;
- **perturbados sinteticamente**;
- **estratificados em subgrupos reais**;
- **controlados**;
- registrados como **metadados**;
- ou classificados como **não simuláveis retrospectivamente**.

Essa decisão evita atribuir significado físico indevido a uma operação de processamento de imagem.

---

# 4. Significado das ações da matriz

## Observar

Inspecionar visualmente ou descrever a distribuição.

## Medir

Extrair uma métrica quantitativa.

## Perturbar

Aplicar transformação sintética controlada.

## Estratificar

Separar exemplos reais por subgrupos.

## Controlar

Garantir equilíbrio, ajuste ou consideração no desenho experimental.

## Não simular

Reconhecer que uma transformação 2D não reproduz adequadamente o fenômeno físico real.

---

# 5. Matriz científica de variáveis

| Fator | Categoria | Observar | Medir | Perturbar | Estratificar | Controlar | Não simular |
|---|---|---:|---:|---:|---:|---:|---:|
| Opacidade/consolidação | Patológica | ✓ | possível | não | possível | ✓ | — |
| Localização da lesão | Patológica | ✓ | possível | não | possível | ✓ | — |
| Anatomia pulmonar | Anatômica | ✓ | possível | não | possível | ✓ | — |
| Intensidade global | Processamento/apresentação | ✓ | ✓ | ✓ | possível | ✓ | — |
| Contraste | Processamento/apresentação | ✓ | ✓ | ✓ | possível | ✓ | — |
| Gamma | Processamento/apresentação | ✓ | ✓ | ✓ | não | ✓ | — |
| Histogram equalization | Processamento | ✓ | ✓ | ✓ como experimento | não | ✓ | — |
| CLAHE | Processamento | ✓ | ✓ | ✓ como experimento | não | ✓ | — |
| Ruído aparente | Qualidade/aquisição | ✓ | ✓ | ✓ com modelo adequado | possível | ✓ | — |
| mAs | Aquisição física | metadado | ✓ se disponível | não diretamente | ✓ | ✓ | ✓ |
| kVp | Aquisição física | metadado | ✓ se disponível | não diretamente | ✓ | ✓ | ✓ |
| Exposure Index | Aquisição física | metadado | ✓ | não | ✓ | ✓ | ✓ |
| DQE/MTF | Detector | contexto | raramente | não | por equipamento | possível | ✓ |
| Detector/fabricante | Domínio | ✓ | categórico | não | ✓ | ✓ | ✓ |
| PA/AP | Geometria | ✓ | categórico | não | ✓ | ✓ | ✓ |
| Upright/supine | Geometria/fisiologia | ✓ | categórico | não | ✓ | ✓ | ✓ |
| Inspiração | Geometria/qualidade | ✓ | possível | não fielmente | ✓ | ✓ | ✓ |
| Rotação do paciente | Geometria | ✓ | possível | não fielmente | ✓ | ✓ | ✓ |
| Rotação 2D | Perturbação computacional | ✓ | ✓ | ✓ | não | ✓ | não confundir com rotação do paciente |
| SID | Geometria | metadado | ✓ se disponível | não | ✓ | ✓ | ✓ |
| Resize | Preprocessamento | ✓ | ✓ | ✓ | não | ✓ | não confundir com projeção |
| Blur | Qualidade | ✓ | ✓ | ✓ | possível | ✓ | não chamar automaticamente de motion blur real |
| Crop | Qualidade/preprocessamento | ✓ | ✓ | ✓ controlado | possível | ✓ | não assumir preservação da evidência |
| Compressão JPEG | Armazenamento | ✓ | ✓ | ✓ | possível | ✓ | — |
| Bit depth | Representação | ✓ | ✓ | possível | ✓ | ✓ | — |
| Window Center/Width | Processamento | ✓ | ✓ | ✓ | possível | ✓ | — |
| LUT | Processamento | ✓ | ✓ | ✓ | possível | ✓ | — |
| MONOCHROME1/2 | Representação | ✓ | categórico | não | ✓ | ✓ | ✓ |
| Texto/marcadores | Artefato | ✓ | possível | possível como controle | ✓ | ✓ | — |
| Cabos/tubos/dispositivos | Artefato/contexto clínico | ✓ | possível | não necessariamente | ✓ | ✓ | — |
| Padding/bordas | Dataset | ✓ | ✓ | possível | ✓ | ✓ | — |
| Hospital | Domínio | ✓ | categórico | não | ✓ | ✓ | ✓ |
| Equipamento | Domínio | ✓ | categórico | não | ✓ | ✓ | ✓ |
| Protocolo institucional | Domínio | ✓ | se disponível | não | ✓ | ✓ | ✓ |
| Prevalência do rótulo | Dataset | ✓ | ✓ | não | ✓ | ✓ | — |
| Label noise | Rótulo | ✓ | estimar | não | ✓ | ✓ | — |

---

# 6. Tipos de experimento derivados da Fase 1

## A. Perturbações sintéticas controladas

Exemplos potenciais:

- intensidade;
- contraste;
- gamma;
- blur;
- ruído;
- downsampling;
- compressão;
- pequenas alterações de apresentação.

Objetivo:

> medir sensibilidade e estabilidade da decisão do modelo.

---

## B. Subgrupos reais

Exemplos:

- PA vs AP;
- portátil vs não portátil;
- hospital;
- equipamento;
- posição;
- metadados de aquisição;
- grupos de qualidade.

Objetivo:

> verificar desempenho diferencial sem fabricar artificialmente uma condição física.

---

## C. Metadados

Exemplos:

- KVP;
- Exposure Index;
- View Position;
- fabricante;
- modelo do equipamento;
- pixel spacing;
- Photometric Interpretation.

Objetivo:

> relacionar condições reais de aquisição ao comportamento do modelo.

---

## D. Limitações não reconstruíveis retrospectivamente

Exemplos:

- transformar fielmente PA em AP;
- reconstruir posição supina a partir de imagem upright;
- reproduzir alteração real de inspiração;
- inferir mAs a partir de brilho final;
- transformar resize em mudança real de SID;
- reproduzir rotação tridimensional do paciente com rotate 2D.

Esses fatores devem ser descritos como:

> variáveis a observar, estratificar ou reconhecer como limitação, e não como augmentations fisicamente equivalentes.

---

# 7. Regras que levaremos para as próximas fases

1. Não chamar intensidade de exposição sem evidência de metadados.
2. Não chamar contraste de simulação de kVp.
3. Não chamar rotate 2D de rotação do paciente.
4. Não chamar resize de simulação de mudança de projeção/SID.
5. Não assumir que HE/CLAHE melhora generalização.
6. Não exigir invariância quando a transformação remove informação clínica.
7. Priorizar DICOM quando isso for compatível com o desenho do estudo.
8. Auditar metadados antes de inferir condições físicas a partir dos pixels.
9. Diferenciar erro do modelo, degradação técnica e possível label noise.
10. Tratar hospital/equipamento/protocolo como potenciais fontes de domain shift e shortcut learning.

---

# 8. Resultado final da Fase 1

A Fase 1 entrega:

- compreensão da cadeia física e digital da radiografia;
- distinção entre patologia e características técnicas;
- reinterpretação da antiga hipótese de luminosidade;
- taxonomia de variáveis;
- regras para perturbações sintéticas;
- critérios para seleção de datasets;
- critérios para auditoria de metadados;
- base para definir o protocolo experimental.

A partir daqui, a próxima fase deve usar essa matriz para avaliar candidatos a datasets e decidir quais variáveis estarão realmente disponíveis em cada base.
