# 1E — Qualidade técnica e interpretação clínica

## Objetivo

Entender quais alterações tornam uma radiografia menos confiável para interpretação e quais podem imitar, ocultar ou modificar a evidência relacionada à pneumonia.

---



## 1. Dimensões de qualidade

Para este projeto, podemos organizar qualidade da CXR em:

- projeção/posicionamento;
- rotação;
- inspiração;
- exposição/ruído;
- cobertura anatômica;
- artefatos;
- movimento.

A pergunta não é apenas:

> “a imagem é boa ou ruim?”

Mas:

> **qual defeito técnico interfere em qual informação clínica?**

---

## 2. Inspiração inadequada

Baixo volume pulmonar pode alterar:

- expansão;
- densidade aparente;
- posição do diafragma;
- sobreposição de estruturas.

Como pneumonia também pode produzir opacidade, existe possibilidade de confundimento.

---

## 3. Rotação

Rotação pode alterar: tornar aparencia enganosa 

- mediastino;
- coração;
- hilos;
- transparência relativa dos pulmões;
- sobreposição.

Assimetria visual pode resultar de:

```text
patologia
ou
geometria inadequada
```

---

## 4. Exposição inadequada

Em radiografia digital, exposição insuficiente pode aumentar ruído.

Exposição elevada pode não produzir uma aparência obviamente “errada” devido ao processamento.

Portanto, não devemos inferir exposição apenas pela aparência.

---

## 5. Movimento

Movimento pode produzir perda de definição e blur.

Um blur sintético pode ser usado para testar sensibilidade à perda de detalhe, mas:

> Gaussian blur não deve ser descrito automaticamente como simulação fiel de movimento radiográfico.

---

## 6. Cobertura anatômica

Uma radiografia pode excluir regiões importantes do pulmão.

Crop agressivo pode:

- remover parte da anatomia;
- remover a própria lesão;
- destruir informação clínica.

Isso significa que:

> o modelo não deve necessariamente ser invariável a qualquer crop.

Se destruirmos a evidência patológica, queda de desempenho pode ser esperada.

---

## 7. Perturbações que preservam versus destroem informação

### Preservação

Transformações leves de apresentação podem manter a evidência clínica.

Nessas situações, esperamos alguma estabilidade.

### Destruição

Exemplos:

- crop agressivo;
- blur extremo;
- compressão extrema;
- saturação extrema.

Nessas situações, a informação clínica pode deixar de existir.

Logo, invariância não é desejável em todos os casos.

---

## 8. Artefatos

Podem existir:

- marcadores;
- letras;
- eletrodos;
- cabos;
- tubos;
- joias;
- roupas;
- dispositivos.

Uma CNN pode aprender correlações com esses elementos.

Portanto:

```text
artefato
  ↓
correlação com rótulo
  ↓
shortcut
```

---

## 9. Ground truth de pneumonia

Pneumonia não é definida perfeitamente apenas pela imagem.

Na prática clínica podem ser considerados:

- radiografia;
- sintomas;
- história;
- exame físico;
- laboratório;
- eventualmente CT.

Logo:

> rótulo de dataset não é sinônimo de verdade clínica perfeita.

---

## 10. Tipos de erro

Ao analisar FP/FN, considerar:

### A. Erro verdadeiro do modelo

Modelo interpretou evidência normal como patológica ou vice-versa.

### B. Problema técnico/artefato

Erro associado à qualidade, ruído, posicionamento ou artefato.

### C. Problema de rótulo

Label original pode estar errado, ambíguo ou representar uma definição operacional diferente.

---

## 11. Nova lógica de análise de erros

Antes:

```text
erro
 ↓
embedding
 ↓
UMAP
 ↓
cluster
```

Agora:

```text
erro
 │
 ├── possível erro ML
 ├── qualidade técnica
 └── possível label noise
        ↓
    análise visual
        ↓
   embedding/UMAP
```

Isso torna a análise mais interpretável.

---

## 12. Qualidade depende da tarefa

Podemos pensar:

```text
Q(imagem, tarefa)
```

Uma imagem pode ser suficiente para detectar uma consolidação grande, mas insuficiente para alterações sutis.

Portanto:

> qualidade não é simplesmente uma variável binária “boa/ruim”.

---

## 13. Quatro conceitos distintos

### Qualidade técnica

O exame foi realizado adequadamente?

### Qualidade da imagem

Existe informação visual suficiente?

### Interpretabilidade clínica

Um profissional consegue avaliar a questão clínica?

### Robustez do modelo

A decisão permanece adequada diante de variações que não deveriam mudar a conclusão?

---

## 14. Conclusões do 1E

1. Qualidade técnica pode alterar ou esconder evidência clínica.
2. Nem toda perturbação deve preservar o diagnóstico.
3. Artefatos podem funcionar como atalhos.
4. FP/FN precisam ser analisados também sob perspectiva de qualidade e label noise.
5. O modelo trabalha com informação visual limitada, não com todo o contexto clínico.



exemplo da cadeia causal 

PACIENTE
│
├── anatomia
└── patologia
       ↓
INTERAÇÃO COM RAIOS X
       ↑
   kVp / mAs
       ↓
   geometria
AP/PA / posição / inspiração / rotação
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
ruído / cobertura / blur / artefatos
       ↓
CONVERSÃO / DATASET
DICOM / PNG / compressão / labels
       ↓
PREPROCESSING ML
       ↓
     MODELO
       ↓
previsão / confiança / erro