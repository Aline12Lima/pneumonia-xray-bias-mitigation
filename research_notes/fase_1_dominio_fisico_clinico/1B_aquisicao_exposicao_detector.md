# 1B — Aquisição, exposição e detector

## Objetivo

Entender quais parâmetros físicos controlam a aquisição da radiografia, como eles alteram o sinal e quais efeitos podem ou não ser reproduzidos de forma válida a partir de uma imagem pronta.

---
Dois parâmetros fundamentais controlados durante a aquisição são:
kVp — tensão do tubo
O kilovoltage peak (kVp) influencia o espectro energético dos raios X e, consequentemente, a capacidade do feixe de atravessar os tecidos.
Simplificando:
kVp ↑ → fótons mais energéticos → maior penetração.
Mas isso também interfere no contraste físico produzido entre estruturas e na radiação espalhada. Por isso não devemos pensar em kVp simplesmente como um controle de "claridade".


## 1. Exposição não é sinônimo de brilho

Precisamos separar:

```text
parâmetros do equipamento
        ↓
exposição do detector
        ↓
dose no paciente
```

Esses conceitos são relacionados, mas não equivalentes.

Em radiografia digital, brilho e contraste podem ser modificados posteriormente pelo processamento.

---

## 2. kVp

O **kVp** influencia o espectro energético dos raios X e a capacidade de penetração.

Simplificando:

```text
kVp ↑
  ↓
fótons mais energéticos
  ↓
maior penetração
```

Também pode alterar relações de contraste e quantidade de radiação espalhada.

Portanto:

> kVp não é um simples controle de “claridade”.

---

## 3. mAs

O **mAs** está relacionado principalmente à quantidade de fótons produzidos.

```text
mAs ↑
 ↓
mais fótons
 ↓
mais sinal
 ↓
menor importância relativa do ruído quântico
 ↓
SNR maior
```

Com mAs menor:

```text
menos fótons
 ↓
maior flutuação estatística relativa
 ↓
quantum noise ↑
 ↓
SNR ↓
```

---

## 4. Quantum noise

A chegada de fótons ao detector é estatística.

Quando existem poucos fótons, as flutuações relativas aumentam.

Por isso, em radiografia digital:

> subexposição tende a aparecer principalmente como aumento de ruído, e não simplesmente como imagem escura.

### Implicação

Uma operação como:

```python
image = image * 0.5
```

não reproduz fisicamente uma redução de mAs.

Ela produz apenas uma transformação de intensidade.

---

## 5. Dose creep

Sistemas digitais possuem ampla faixa dinâmica e processamento capaz de compensar parte da aparência visual.

Isso pode fazer com que exposições elevadas ainda produzam imagens visualmente aceitáveis, contribuindo para o fenômeno conhecido como **dose creep**.

Logo:

> brilho da imagem final não é um indicador confiável da exposição.

---

## 6. Exposure Index, Target EI e Deviation Index

Quando disponíveis, sistemas digitais podem registrar:

- **Exposure Index (EI)**;
- **Target Exposure Index (EI_T)**;
- **Deviation Index (DI)**.

Esses metadados são muito mais informativos sobre exposição real do que inferir condições de aquisição pela aparência visual.

---

## 7. Detector

Depois de atravessar o paciente, a radiação chega ao detector.

Podem existir diferentes tecnologias, por exemplo:

- CR;
- DR;
- conversão direta;
- conversão indireta.

A tecnologia do detector influencia:

- eficiência;
- resolução espacial;
- ruído;
- resposta do sistema.

---

## 8. DQE e MTF

### DQE — Detective Quantum Efficiency

Representa, de forma simplificada, a eficiência com que o detector transforma os fótons recebidos em informação útil.

### MTF — Modulation Transfer Function

Relaciona-se à capacidade do sistema de preservar detalhes espaciais em diferentes frequências.

Dois equipamentos diferentes podem, portanto, produzir imagens com características diferentes mesmo sob condições clínicas semelhantes.

---

## 9. SNR e CNR

### SNR

```text
Signal-to-Noise Ratio
```

Representa quanto o sinal útil se destaca do ruído.

### CNR

```text
Contrast-to-Noise Ratio
```

Relaciona a diferença entre sinais de regiões distintas ao nível de ruído.

Essas métricas ajudam a caracterizar qualidade física, mas não equivalem automaticamente a desempenho diagnóstico.

---

## 10. Scatter

Radiação espalhada pode:

- reduzir contraste;
- degradar SNR;
- alterar a qualidade da imagem.

Grades antiespalhamento podem ser utilizadas para reduzir esse efeito.

---

## 11. DICOM versus PNG/JPEG

### Se houver DICOM

Podemos potencialmente acessar:

- KVP;
- exposição;
- tempo;
- corrente;
- fabricante;
- modelo;
- detector;
- EI;
- pixel spacing;
- photometric interpretation.

### Se houver apenas PNG/JPEG

Podemos medir:

- intensidade;
- histograma;
- contraste;
- entropia;
- ruído aparente;
- sharpness.

Mas não devemos inferir diretamente:

- mAs;
- kVp;
- exposição real.

---

## 12. Conclusões do 1B

1. Brilho digital não deve ser usado como sinônimo de exposição.
2. mAs e kVp representam mecanismos físicos diferentes.
3. Subexposição está fortemente relacionada a ruído/SNR.
4. Detector influencia eficiência, ruído e resolução.
5. DICOM preserva potencialmente informações importantes de aquisição.
6. Transformações de intensidade não devem ser apresentadas como simulações físicas de exposição.

---

## 13. Implicação metodológica

| Fenômeno | Transformação simples | Interpretação |
|---|---|---|
| mAs menor | brilho menor | incorreta |
| mAs menor | ruído sintético | aproximação possível |
| kVp diferente | contraste global | apenas aproximação |
| detector diferente | resize | não equivalente |
| resolução degradada | blur/downsampling | aproximação controlada |
| quantum noise | modelo adequado de ruído | potencialmente defensável |

A pergunta experimental precisa sempre distinguir:

> **robustez a uma transformação digital**

de

> **simulação de uma condição física real**.
## Modelo causal de aquisição 
              AQUISIÇÃO
                  │
       ┌──────────┴───────────┐
       ↓                      ↓
      kVp                    mAs
       │                      │
energia/espectro        nº de fótons
penetração                   │
contraste físico             ↓
       │                 quantum noise
       │                      │
       └──────────┬───────────┘
                  ↓
               PACIENTE
                  │
        atenuação + scatter
                  │
                  ↓
              DETECTOR
                  │
          ┌───────┴────────┐
          ↓                ↓
         DQE              MTF
          │                │
          ↓                ↓
        SNR/ruído      resolução/detalhe
          │                │
          └───────┬────────┘
                  ↓
             sinal digital
                  │
                  ↓
         PÓS-PROCESSAMENTO
          [será a Fase 1D]