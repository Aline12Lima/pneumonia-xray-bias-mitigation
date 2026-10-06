# 1D — Pós-processamento e formação da imagem digital final

## Objetivo

Entender como o sinal capturado pelo detector é transformado até chegar à imagem utilizada pelo modelo.

Esta etapa é central para reinterpretar corretamente o conceito de “luminosidade”.

---

## 1. Cadeia digital

```text
detector
   ↓
sinal bruto
   ↓
correções do detector
   ↓
For Processing
   ↓
processamento do fabricante
   ↓
LUT / windowing
   ↓
contraste / sharpness / redução de ruído
   ↓
For Presentation
   ↓
conversão DICOM → PNG/JPEG
   ↓
preprocessamento ML
   ↓
modelo
```

---

## 2. For Processing versus For Presentation

### For Processing

Representação destinada a processamento posterior.


### For Presentation

Imagem preparada para visualização/interpretação.

O sistema pode alterar:

- contraste;
- brilho;
- nitidez;
- ruído;
- resposta local da imagem.

Portanto, a distribuição final dos pixels não depende apenas da aquisição física.

---

## 3. Window Center e Window Width

Windowing seleciona uma faixa de valores de interesse para apresentação.

- **Window Center**: região central.
- **Window Width**: largura da faixa.

Dois windowings diferentes podem produzir aparências muito diferentes a partir dos mesmos valores originais.

Logo:

> mudança de apresentação não significa mudança de aquisição.

---

## 4. LUT

Uma **Look-Up Table** transforma valores de entrada em valores de saída.

A transformação pode ser não linear.

Isso significa que:

```text
mesma imagem adquirida
        ↓
LUT A → aparência A
LUT B → aparência B
```

---

## 5. MONOCHROME1 e MONOCHROME2

DICOM pode definir diferentes interpretações fotométricas.

### MONOCHROME1

Valores menores são exibidos mais claros.

### MONOCHROME2

Valores menores são exibidos mais escuros.

Esse campo precisa ser respeitado na leitura.

Uma interpretação incorreta pode inverter completamente a aparência da radiografia.

---

## 6. Processamento de contraste e detalhe

Sistemas podem aplicar:

- gradation processing;
- processamento multifrequencial;
- edge enhancement;
- noise reduction;
- equalização;
- processamento local.

Essas operações podem alterar significativamente:

- contraste;
- textura;
- sharpness;
- intensidade;
- aparência regional.

---

## 7. Histogram Equalization e CLAHE

Técnicas como HE e CLAHE redistribuem intensidades para aumentar contraste.

### Cuidado

Não devemos assumir que “melhor contraste” implica automaticamente “melhor generalização”.

Essas técnicas podem:

- modificar a distribuição dos dados;
- remover pistas;
- criar novo domínio;
- alterar desempenho externo.

Portanto, HE/CLAHE devem ser avaliados experimentalmente, não aplicados por padrão.

---

## 8. Conversão DICOM → PNG/JPEG

Essa conversão pode envolver:

- windowing;
- clipping;
- normalização;
- LUT;
- inversão;
- quantização;
- redução de bit depth.

Exemplo conceitual:

```text
DICOM 12 bits
0 ... 4095

        ↓

PNG 8 bits
0 ... 255
```

A conversão não é necessariamente neutra.

---

## 9. Pré-processamento do pesquisador

Depois da imagem final ainda podem ser aplicados:

- resize;
- normalização;
- divisão por 255;
- ImageNet normalization;
- CLAHE;
- brightness augmentation;
- contrast augmentation;
- noise;
- blur.

Precisamos separar:

### pós-processamento radiográfico

realizado pelo sistema/equipamento.

### pré-processamento computacional

realizado pelo pipeline de ML.

---

## 10. Reinterpretação da luminosidade

No estudo anterior:

```text
imagem
 ↓
luminosidade
 ↓
erro do modelo
```

Agora:

```text
aquisição
 ↓
detector
 ↓
processamento
 ↓
conversão
 ↓
preprocessing
 ↓
distribuição de intensidade
 ↓
modelo
```

Logo:

> intensidade observada é uma variável downstream de múltiplas causas.

---

## 11. Brightness augmentation continua válido?

Sim, mas a interpretação precisa ser correta.

Uma transformação:

```text
I' = αI
```

pode testar:

> sensibilidade do modelo à intensidade global.

Não deve ser apresentada como:

> simulação de subexposição ou alteração de mAs.

---

## 12. Contraste

Mudanças de contraste podem testar:

> sensibilidade do modelo a alterações controladas de contraste.

Mas não devem ser descritas automaticamente como:

> simulação de kVp.

---

## 13. Conclusões do 1D

1. A aparência final depende fortemente do processamento.
2. Windowing e LUT alteram apresentação sem modificar aquisição.
3. MONOCHROME1/2 precisam ser tratados corretamente.
4. DICOM → PNG/JPEG faz parte do protocolo científico.
5. HE/CLAHE não devem ser assumidos como benéficos.
6. Brilho/intensidade é característica observável da imagem, não proxy direto de exposição.

---

## 14. Implicação metodológica

Podemos separar dois experimentos:

### Robustez de apresentação

```text
imagem original
      ↓
transformação controlada
      ↓
medir estabilidade da previsão
```

### Generalização de domínio

```text
treino em hospital/dataset A
      ↓
teste em B/C
```

São análises complementares, mas respondem perguntas distintas.
