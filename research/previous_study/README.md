# Estudo anterior — arquivo histórico

Versão anterior da pesquisa, baseada em DenseNet121/CBAM, clustering, UMAP e investigação de viés de luminosidade. Preservada para rastreabilidade e comparação; seus resultados não são assumidos como conclusões do novo protocolo.

- Notebook: [ic-usf-classification-pneumonia-clustering-v2.ipynb](notebooks/ic-usf-classification-pneumonia-clustering-v2.ipynb).
- Modelo local: `models/modelo_B_cbam_best.h5` (não versionado no Git).
- Código original: `src/analysis/`, `src/models/` e `src/training/`.

O notebook, o modelo e os códigos foram preservados sem alterações de conteúdo.
O arquivo `src/training/train.py` já estava incompleto no projeto original (termina durante a chamada de `flow_from_directory`).
Os imports `src.models` desse código pressupõem que a raiz de execução seja `research/previous_study/`.
Caminhos e dependências do notebook histórico precisam ser revisados antes de uma nova execução.
Dados locais anteriormente em `data/train`, `data/val` e `data/test` agora ficam em `research/previous_study/data/`, separados dos datasets da nova pesquisa.

A documentação original abaixo é histórica: suas afirmações não foram revalidadas nesta reorganização.
O arquivo `docs/paper.pdf` mencionado não estava presente, e o link de reprodução original contém um marcador provisório.

---

# Mitigation of Luminosity Bias in Chest X-Ray Pneumonia Classification

Deep Learning with DenseNet121 and CBAM for robust and interpretable medical image classification
Kaggle - https://www.kaggle.com/code/aline123lima/ic-usf-classification-pneumonia-clustering-v2

## 🧠 Problem
Deep learning models applied to chest X-ray pneumonia classification often achieve high accuracy
by exploiting spurious correlations, such as image brightness and contrast, rather than true
pulmonary morphology. This phenomenon, known as shortcut learning, can lead to unreliable
clinical behavior and poor generalization.

## 🔬 Approach

This project investigates luminosity bias in convolutional neural networks by comparing a
DenseNet121 baseline model with a DenseNet121 architecture enhanced with the Convolutional
Block Attention Module (CBAM).

Beyond standard performance metrics, the models are analyzed through latent space topology
using UMAP and HDBSCAN, enabling a detailed investigation of internal representations,
error distribution, and shortcut learning behavior.

## 🧪 Experiments

- Dataset: Chest X-Ray Pneumonia Dataset (Kermany et al.)
- Input size: 224 × 224
- Architectures:
  - DenseNet121 (baseline)
  - DenseNet121 + CBAM
- Optimization:
  - Adam optimizer
  - Binary Crossentropy loss
  - Early Stopping and ReduceLROnPlateau
- Evaluation:
  - Accuracy, Sensitivity
  - False Positives and False Negatives analysis

## 📊 Key Results

- The baseline model formed only 3 brightness-driven macro-clusters in the latent space.
- The CBAM-enhanced model reorganized representations into 47 morphology-based micro-clusters.
- Luminosity bias was significantly reduced, leading to clinically coherent error patterns.
- Remaining errors were associated with subtle opacities and low-contrast cases rather than
  acquisition artifacts.
## 🧠 Interpretability and Error Analysis

Latent space projections using UMAP and clustering with HDBSCAN revealed how attention
mechanisms reshape internal representations. Errors were mapped onto micro-clusters,
demonstrating that the CBAM model fails primarily on challenging clinical cases rather than
on spurious visual cues.
## 📄 Scientific Paper

The full paper describing the methodology, experiments, and results is available in:

📄 `docs/paper.pdf`
## 🔗 Reproducibility

All experiments were executed on Kaggle due to GPU availability.
The complete and reproducible notebook is available at:

🔗 https://www.kaggle.com/SEU-LINK-AQUI

## 🧰 Tools and Technologies

- Programming Language: Python
- Deep Learning: TensorFlow / Keras
- Architectures: DenseNet121, CBAM
- Representation Analysis: UMAP, HDBSCAN
- Data Processing: NumPy, Pandas
- Visualization: Matplotlib, Seaborn
- Environment: Kaggle, Jupyter Notebook
- Version Control: Git, GitHub
## 🚀 Future Work

- Attention ablation studies (SE, ECA)
- Robustness analysis under illumination perturbations
- Extension to multi-center datasets
- Deployment-oriented inference analysis
