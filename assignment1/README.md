# Pattern Recognition Assignments

Este repositório contém as resoluções das atividades da disciplina **Reconhecimento de Padrões**, do mestrado na UTFPR.

## 📌 Atividade 1 – Modelo ADALINE com Wine Dataset

- Foi utilizado o dataset *Wine* da UCI Machine Learning Repository.
- O modelo ADALINE foi treinado com o `SGDClassifier` da biblioteca `scikit-learn`.
- Foram testadas diferentes combinações de atributos para avaliação visual.
- O melhor par de atributos encontrado foi:
  - **Flavanoids** e **OD280/OD315 of diluted wines**

### 📊 Resultados
Os gráficos e a análise completa estão disponíveis no arquivo [`adaline_assignment1_wine.pdf`](adaline_assignment1_wine.pdf)

---

## ▶️ Execução

Requisitos:
```bash
pip install -r requirements.txt
```

Execução:
```bash
python adaline_wine_analysis.py
```

---

## 📁 Arquivos

- `adaline_wine_analysis.py`: código-fonte da atividade 1
- `wine.data`: arquivo de dados
- `adaline_assignment1_wine.pdf`: relatório com resultados