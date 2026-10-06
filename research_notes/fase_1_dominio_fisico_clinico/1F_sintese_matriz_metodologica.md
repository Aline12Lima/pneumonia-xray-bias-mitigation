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


A matriz abaixo organiza os fatores identificados na Fase 1 de acordo com a forma como eles devem ser tratados no projeto.

As ações possíveis são:

- **Observar** → inspecionar visualmente ou descrever.
- **Medir** → extrair quantitativamente uma característica.
- **Perturbar** → aplicar uma transformação sintética controlada.
- **Estratificar** → separar exemplos reais em subgrupos.
- **Controlar** → considerar explicitamente no desenho experimental.
- **Não simular** → reconhecer que uma transformação 2D não representa adequadamente o fenômeno físico real.

---

## 5.1 Variáveis patológicas e anatômicas

### Opacidade / consolidação
- **Categoria:** Patológica
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não se aplica
- **Interpretação:** representa potencial evidência clínica que o modelo pode legitimamente utilizar.

### Localização da lesão
- **Categoria:** Patológica
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não se aplica
- **Interpretação:** pode ser utilizada para verificar se erros ou sensibilidades se concentram em determinadas regiões pulmonares.

### Anatomia pulmonar
- **Categoria:** Anatômica
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não se aplica
- **Interpretação:** informação anatômica real que deve ser preservada durante o preprocessing.

---

## 5.2 Intensidade e processamento da imagem

### Intensidade global
- **Categoria:** Processamento / apresentação
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** pode ser usada em testes de robustez à intensidade global.
- **Cuidado:** não interpretar automaticamente como exposição radiográfica.

### Contraste
- **Categoria:** Processamento / apresentação
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** alteração sintética de contraste não deve ser descrita automaticamente como alteração de kVp.

### Gamma
- **Categoria:** Processamento / apresentação
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** não prioritariamente
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** transformação não linear de intensidade útil para testar sensibilidade à apresentação.

### Histogram Equalization
- **Categoria:** Processamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** apenas como experimento controlado
- **Estratificar:** não
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** não assumir que melhora automaticamente a generalização.

### CLAHE
- **Categoria:** Processamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** apenas como experimento controlado
- **Estratificar:** não
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** pode alterar localmente contraste e distribuição dos dados.

### Window Center / Window Width
- **Categoria:** Processamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** representa transformação de apresentação, não nova aquisição.

### LUT
- **Categoria:** Processamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** importante para entender diferenças entre imagem de processamento e imagem apresentada.

---

## 5.3 Aquisição física

### mAs
- **Categoria:** Aquisição física
- **Observar:** via metadado
- **Medir:** sim, se disponível
- **Perturbar:** não diretamente
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** afeta quantidade de fótons, ruído e SNR.
- **Cuidado:** reduzir brightness não equivale a reduzir mAs.

### kVp
- **Categoria:** Aquisição física
- **Observar:** via metadado
- **Medir:** sim, se disponível
- **Perturbar:** não diretamente
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** influencia espectro energético, penetração e contraste físico.
- **Cuidado:** alterar contraste digital não equivale a alterar kVp.

### Exposure Index
- **Categoria:** Aquisição física
- **Observar:** via metadado
- **Medir:** sim
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** quando disponível, é preferível à inferência visual de exposição.

### Ruído aparente
- **Categoria:** Qualidade / aquisição
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim, com modelo de ruído adequado
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não necessariamente
- **Cuidado:** o ruído sintético deve ser descrito como aproximação, não como reprodução perfeita da aquisição.

---

## 5.4 Detector e equipamento

### DQE / MTF
- **Categoria:** Detector
- **Observar:** como contexto técnico
- **Medir:** raramente, dependendo dos dados disponíveis
- **Perturbar:** não
- **Estratificar:** por equipamento, se possível
- **Controlar:** possível
- **Não simular:** sim
- **Interpretação:** caracterizam eficiência e transferência de detalhe do sistema.

### Detector / fabricante
- **Categoria:** Domínio
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** possível fonte de domain shift e shortcut learning.

### Equipamento
- **Categoria:** Domínio
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim

---

## 5.5 Geometria da aquisição

### PA / AP
- **Categoria:** Geometria
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** deve ser avaliada com exemplos reais, não fabricada por transformação 2D.

### Upright / supine
- **Categoria:** Geometria / fisiologia
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim

### Inspiração
- **Categoria:** Geometria / qualidade
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não fielmente
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** altera volume pulmonar e aparência das estruturas.

### Rotação do paciente
- **Categoria:** Geometria
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não fielmente
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Cuidado:** rotate 2D não equivale à rotação tridimensional do paciente.

### SID
- **Categoria:** Geometria
- **Observar:** via metadado
- **Medir:** sim, se disponível
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim

---

## 5.6 Perturbações computacionais e qualidade

### Rotação 2D
- **Categoria:** Perturbação computacional
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** não
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** não descrever como rotação real do paciente.

### Resize
- **Categoria:** Preprocessamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** não
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** não interpretar como alteração de projeção ou SID.

### Blur
- **Categoria:** Qualidade
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** parcialmente
- **Cuidado:** Gaussian blur não deve ser chamado automaticamente de motion blur real.

### Crop
- **Categoria:** Qualidade / preprocessamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim, de forma controlada
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Cuidado:** crop pode remover informação patológica; portanto, invariância não é necessariamente desejável.

### Compressão JPEG
- **Categoria:** Armazenamento
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** sim
- **Estratificar:** possível
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** útil para testar robustez a perda/compressão de armazenamento.

---

## 5.7 Representação digital

### Bit depth
- **Categoria:** Representação
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** possível
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não

### MONOCHROME1 / MONOCHROME2
- **Categoria:** Representação
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** deve ser tratado corretamente na leitura do DICOM.

---

## 5.8 Artefatos e elementos externos

### Texto / marcadores
- **Categoria:** Artefato
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** possível como controle experimental
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** possível fonte de shortcut learning.

### Cabos / tubos / dispositivos
- **Categoria:** Artefato / contexto clínico
- **Observar:** sim
- **Medir:** possível
- **Perturbar:** não necessariamente
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** podem também funcionar como proxies de gravidade ou ambiente hospitalar.

### Padding / bordas
- **Categoria:** Dataset
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** possível
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não

---

## 5.9 Domínio e instituição

### Hospital
- **Categoria:** Domínio
- **Observar:** sim
- **Medir:** categórico
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim
- **Interpretação:** uma das principais variáveis para avaliar generalização externa.

### Protocolo institucional
- **Categoria:** Domínio
- **Observar:** sim
- **Medir:** se disponível
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** sim

---

## 5.10 Rótulos e composição do dataset

### Prevalência do rótulo
- **Categoria:** Dataset
- **Observar:** sim
- **Medir:** sim
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** deve ser considerada nas comparações entre datasets.

### Label noise
- **Categoria:** Rótulo
- **Observar:** sim
- **Medir:** estimar quando possível
- **Perturbar:** não
- **Estratificar:** sim
- **Controlar:** sim
- **Não simular:** não
- **Interpretação:** FP/FN podem refletir erro do modelo, ambiguidade clínica ou erro do rótulo.

---

# 5.11 Síntese operacional

A Fase 1 indica que as variáveis devem seguir quatro estratégias principais:

### Perturbações sintéticas controladas

Usar principalmente para testar robustez a:

- intensidade;
- contraste;
- gamma;
- ruído;
- blur;
- downsampling;
- compressão;
- transformações de apresentação.

### Subgrupos reais

Usar para:

- PA/AP;
- posição;
- inspiração;
- hospital;
- equipamento;
- dispositivos;
- condições reais de aquisição.

### Metadados

Priorizar quando disponíveis:

- kVp;
- mAs;
- Exposure Index;
- View Position;
- Patient Position;
- fabricante;
- equipamento;
- Pixel Spacing;
- Photometric Interpretation.

### Não simular retrospectivamente

Não tratar como equivalentes físicos:

- brightness ≠ exposição;
- contraste ≠ kVp;
- rotate 2D ≠ rotação do paciente;
- resize ≠ mudança de SID;
- PA ≠ AP sintético;
- upright ≠ supine sintético;
- baixa inspiração ≠ simples resize/crop.

---

## 5.12 Regra metodológica final

> **Não vamos transformar todos os fatores identificados em augmentations.**

Alguns serão **perturbações sintéticas controladas**, outros serão estudados como **subgrupos reais**, alguns entrarão como **metadados ou variáveis de controle**, e outros serão reconhecidos explicitamente como **limitações que não podem ser reconstruídas retrospectivamente a partir da imagem final**.
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
