# Pattern Recognition Assignments

Este repositório contém as resoluções das atividades da disciplina **Reconhecimento de Padrões**, do mestrado na UTFPR.

---

## 📌 Atividade 1 – Modelo ADALINE com Wine Dataset

- Utilizado o dataset *Wine* da UCI.
- Modelo ADALINE treinado com `SGDClassifier`.
- Avaliação visual com diferentes pares de atributos.
- Melhor desempenho visual com: **Flavanoids** e **OD280/OD315 of diluted wines**
- 📄 [`assignment1/adaline_assignment1_wine.pdf`](assignment1/adaline_assignment1_wine.pdf)
- 📜 Código: [`assignment1/adaline_wine_analysis.py`](assignment1/adaline_wine_analysis.py)

---

## 📌 Atividade 2 – Previsão de Sobrevivência no Titanic (Kaggle)

- Utilizado os arquivos `train.csv` e `test.csv` do desafio Titanic (Kaggle).
- Modelo: `RandomForestClassifier`
- Pré-processamento:
  - Preenchimento de nulos
  - Extração de título (`Title`) dos nomes
  - Criação do atributo `FamilySize`
- Validação cruzada com acurácia média de **~80.7%**
- 📄 [`assignment2/assignment2_titanic_report.pdf`](assignment2/assignment2_titanic_report.pdf)
- 📂 Arquivo de submissão: [`assignment2/submission_assignment2.csv`](assignment2/submission_assignment2.csv)
- 📜 Código: [`assignment2/assignment2_titanic_model.py`](assignment2/assignment2_titanic_model.py)

---

## ▶️ Execução

Instale os requisitos:
```bash
pip install -r requirements.txt
```

Execute os scripts:
```bash
python assignment1/adaline_wine_analysis.py
python assignment2/assignment2_titanic_model.py
```