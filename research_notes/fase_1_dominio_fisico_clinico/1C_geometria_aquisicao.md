# 1C — Geometria da aquisição

## Objetivo

Entender como projeção, posicionamento, rotação, inspiração e distância alteram a anatomia projetada e por que essas variações não devem ser tratadas como simples transformações 2D.

---

## 1. Radiografia é uma projeção 2D de uma anatomia 3D

A imagem depende da geometria entre:

- fonte;
- paciente;
- estruturas internas;
- detector.

Logo:

> mudar a posição do paciente antes da aquisição não equivale a editar uma radiografia pronta.

---

## 2. PA versus AP

### PA

Feixe:

```text
posterior → anterior
```

### AP

Feixe:

```text
anterior → posterior
```

A diferença de projeção altera a magnificação e a relação entre estruturas e detector.

O coração, por exemplo, pode parecer maior em AP.

---

## 3. Magnificação e SID

A magnificação depende da geometria da projeção.

O **SID — Source-to-Image Distance** também influencia a representação geométrica.

Portanto:

```text
PA/AP
+
posição
+
SID
+
distância estrutura-detector
=
geometria final
```

Um simples `resize()` não recria esse fenômeno.

---

## 4. Upright versus supine

A posição corporal pode alterar:

- geometria;
- expansão pulmonar;
- distribuição de fluidos;
- visualização de pneumotórax e derrame;
- relação entre estruturas.

Não existe uma transformação 2D simples capaz de transformar fielmente uma imagem upright em supine.

---

## 5. Inspiração

A inspiração modifica o volume pulmonar e a relação espacial entre estruturas.

Baixa inspiração pode produzir:

- pulmões menos expandidos;
- aumento aparente de densidade;
- maior sobreposição de estruturas.

Isso é especialmente relevante porque pneumonia também pode produzir aumento de opacidade.

Portanto:

```text
baixa inspiração
        ↓
mudança de aparência pulmonar
        ↓
possível confundimento
```

---

## 6. Rotação do paciente

Rotação real altera a projeção de estruturas tridimensionais.

Pode mudar:

- mediastino;
- hilos;
- assimetria pulmonar;
- sobreposição;
- visibilidade de opacidades.

### Importante

Rotacionar a imagem em 2D:

```python
rotate(image, 10)
```

não equivale a rotacionar o paciente.

Podemos representar isso como:

```text
projetar(anatomia rotacionada)
≠
rotacionar(projeção pronta)
```

---

## 7. Confundimento por projeção

AP pode ser mais frequente em:

- pacientes hospitalizados;
- pacientes graves;
- pacientes acamados;
- exames portáteis.

PA pode ser mais frequente em:

- pacientes ambulatoriais;
- pessoas capazes de permanecer em pé.

Logo, é possível surgir:

```text
AP
 ↓
paciente mais grave
 ↓
maior prevalência de doença
 ↓
modelo aprende AP como shortcut
```

---

## 8. Variáveis a auditar

Quando disponíveis:

- View Position;
- Patient Position;
- Portable;
- PA/AP;
- lateral;
- rotação;
- inspiração;
- SID.

Essas variáveis podem ser cruzadas com o rótulo.

Exemplo:

```text
P(AP | pneumonia)
vs.
P(AP | normal)
```

---

## 9. Decisão metodológica

| Fator | Simular? | Estratégia |
|---|---|---|
| PA/AP | não | estratificar por exemplos reais |
| upright/supine | não | estratificar |
| inspiração | não fielmente | auditar |
| rotação do paciente | não | auditar |
| SID | não fielmente | metadado/limitação |
| rotação 2D da imagem | sim | teste de robustez, sem chamar de rotação do paciente |
| crop | sim | teste controlado |
| resize | sim | teste de resolução/escala |

A principal conclusão é:

> variação real de aquisição e perturbação sintética são categorias diferentes.

## detalhes 
AP costuma ser realizada em quem?
Frequentemente:
pacientes hospitalizados, graves, acamados, UTI. PubMed Central (PMC)
PA?
Frequentemente:
pacientes capazes de ficar em pé e realizar exame convencional
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Agora imagine nosso dataset:
PNEUMONIA
80% AP

NORMAL
80% PA
Para cada imagem, quando os dados permitirem:
Projection:
PA / AP / lateral

Patient position:
upright / supine / semi-erect

Rotation:
adequada / inadequada

Inspiration:
adequada / inadequada

SID:
se disponível

Portable:
sim / não

Depois cruzaremos isso com:
pneumonia
normal

Por exemplo:
\[
P(AP\mid pneumonia)
\]


 ## RESULTADO !! 
 
PERTUBAÇÃO sINTÉTICA É DIFERENTE DE VARIAÇÃO REAL DE AQUISIÇÃO 

para fenômenos como PA/AP, supine/upright, inspiração e rotação do paciente, nossa estratégia principal deverá ser procurar exemplos reais/metadados, e não tentar fabricar artificialmente a condição.
Isso também começa a definir um critério para escolher nossos três datasets: além de preferirmos DICOM quando possível, precisamos verificar se existem metadados de View Position / Patient Position / aquisição portátil ou alguma forma confiável de estratificar essas condições.        