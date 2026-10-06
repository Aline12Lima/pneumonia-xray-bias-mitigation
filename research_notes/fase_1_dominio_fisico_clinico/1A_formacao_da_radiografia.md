# 1A — Formação da radiografia e origem das características da imagem
O tubo de raios X produz radiação. Dois parâmetros importantes são:
kVp — kilovoltage peak
mAs — milliampere-seconds

EXPOSIÇÃO RADIOGRÁFICA
       │
       ├── kVp
       │     ↓
       │   energia/espectro
       │   penetração
       │   relações de contraste
       │
       └── mAs
             ↓
          nº de fótons
             ↓
          ruído / SNR

## Objetivo

Entender de onde vêm as características visuais de uma radiografia de tórax antes de atribuir qualquer alteração da imagem à doença, à exposição ou ao processamento.

---

## 1. Cadeia física básica

A radiografia de tórax resulta da seguinte cadeia:

```text
emissão de raios X
        ↓
interação com tecidos
        ↓
atenuação
        ↓
exposição do detector
        ↓
aquisição digital
        ↓
pós-processamento
        ↓
imagem apresentada
        ↓
interpretação clínica
```

A intensidade de um pixel não representa simplesmente “luminosidade”. Ela é consequência de múltiplas etapas físicas e computacionais.

---

## 2. Interação dos raios X com o corpo

Diferentes estruturas atenuam os raios X de forma diferente.

Simplificando:

- **ar**: pouca atenuação, tende a aparecer mais escuro;
- **tecidos moles**: atenuação intermediária;
- **ossos e estruturas mais densas**: maior atenuação, tendem a aparecer mais claros.

O tórax contém uma grande faixa de atenuações: pulmões, vasos, coração, mediastino, costelas, coluna e diafragma.

Portanto, parte importante da distribuição de intensidades é determinada pela **anatomia real do paciente**.

---

## 3. Onde entra a pneumonia

A pneumonia pode produzir alterações como:

- opacidades;
- consolidações;
- alterações focais ou difusas;
- mudanças na densidade radiográfica pulmonar.

A cadeia de interesse é:

```text
patologia
   ↓
alteração do tecido pulmonar
   ↓
alteração da atenuação
   ↓
padrão radiográfico
   ↓
evidência potencialmente útil para o modelo
```

Essa é a informação que queremos que o modelo utilize.

---

## 4. O problema: a aparência não depende apenas da doença

Duas radiografias de pacientes com condição clínica semelhante podem apresentar distribuições de pixels diferentes por causa de:

- kVp;
- mAs;
- detector;
- posicionamento;
- projeção;
- inspiração;
- processamento;
- equipamento;
- resolução;
- compressão;
- artefatos;
- protocolo institucional.

Isso significa que:

> **mesma patologia não implica mesma distribuição de pixels.**

---

## 5. Três grandes categorias iniciais

### A. Patológica/anatômica

Informação que pode ser legitimamente utilizada pelo modelo:

- opacidades;
- consolidações;
- distribuição das alterações;
- anatomia pulmonar;
- estruturas relevantes ao diagnóstico.

### B. Técnica/de aquisição

Características que alteram a imagem sem representar diretamente pneumonia:

- exposição;
- projeção;
- posição;
- inspiração;
- detector;
- resolução;
- ruído;
- processamento.

### C. Artefato/dataset/domínio

Características idealmente não utilizadas como atalho:

- textos;
- marcadores;
- bordas;
- padding;
- compressão específica;
- hospital de origem;
- equipamento;
- padrões do workflow;
- diferenças institucionais.

---

## 6. Implicação para ML

Uma CNN não sabe automaticamente distinguir:

```text
evidência clínica
vs.
característica técnica
vs.
artefato
```

Ela aprende correlações presentes no conjunto de treinamento.

Logo, um modelo pode aprender:

```text
característica do hospital
        ↓
correlação com o rótulo
        ↓
"pneumonia"
```

em vez de:

```text
alteração pulmonar
        ↓
pneumonia
```

Esse comportamento é compatível com **shortcut learning**.

---

## 7. Consequência para a antiga hipótese de luminosidade

A formulação:

> “luminosidade causa viés”

é excessivamente simples.

A formulação mais defensável é:

> características de aquisição, processamento e apresentação podem produzir variações de aparência que influenciam a decisão do modelo.

A intensidade/luminosidade passa a ser **uma variável investigada**, não a explicação inteira.

---

## 8. Saída conceitual do 1A

A pergunta do projeto passa a ser:

> Quanto a decisão do modelo permanece dependente da evidência clínica quando variam características técnicas, de aquisição, processamento e domínio que não representam diretamente a patologia?

Essa pergunta orienta as demais partes da Fase 1.


continuação do estudo...
## 9. Depois os raios X encontram o paciente
Diferentes estruturas atenuam os raios X de formas diferentes.
AR
↓ pouca atenuação
↓ muitos fótons chegam ao detector
↓
região escura

TECIDOS MOLES
↓
atenuação intermediária

OSSO / estruturas densas
↓ muita atenuação
↓ menos fótons chegam ao detector
↓
região clara

O tórax é particularmente complicado porque contém uma enorme faixa de atenuações: pulmões cheios de ar, vasos, coração, mediastino, coluna, costelas etc. Isso exige grande faixa dinâmica do sistema de radiografia (Digital chest radiography: an update on modern technology, dose containment and control of image quality)

## 10. A Variável Clínica 
Pneumonia 

Pneumonia pode produzir alterações radiográficas como opacidades/consolidações. Dependendo do tipo e do quadro, localização, extensão e aparência podem variar; achados radiográficos também não devem ser interpretados isoladamente do contexto clínico.(Guidelines for diagnosis and management of community- and hospital-acquired pneumonia in adults: Joint ICS/NCCP(I) recommendations)

A mesma anatomia pode produzir iamgens visualmente diferentes, porque depois que o raio-x
atravessa o corpo ainda existe fontes de variação como:
Exposição --->  Detector --> Fabricante --> Processamentos --> LUT/windowing --> Enhancement --> Armazenamento

Os estudos na(New developed DR detector performs radiographs of hand, pelvic and premature chest anatomies at a lower radiation dose and/or a higher image quality), comparam sistemas digitais se encontram diferenças mensuraveis entre detectores....e detectores diferentes possuem propriedade distintas de eficiência, ruidp e resolução

No pos-processamento: Os valores originalmente capturados pelo detector não precisam corresponder diretamente aos pixels que veremos no PNG/JPEG usado no treinamento.

sinal físico capturado
        ↓
imagem bruta
        ↓
correções
        ↓
processamento
        ↓
LUT / grayscale
        ↓
contraste / detalhes / ruído
        ↓
imagem apresentada

## A radiografia dependende como os pacientes foram examinados: 
Fatores:rotação;

inspiração;
posicionamento;
centralização;
colimação;
projeção;
partes anatômicas cortadas;
objetos externos;
movimento.

característica técnica/de aquisição

kVp
mAs
projeção
posição
inspiração
rotação
detector
resolução
ruído
processamento

O Dataset:

hospital
equipamento
processamento
compressão
marcadores
prevalência
população
protocolo

Hospital A
    ↓
equipamento A
processamento A
protocolo A
população A
marcadores A
prevalência A

Hospital B
    ↓
equipamento B
processamento B
protocolo B
população B
marcadores B
prevalência B