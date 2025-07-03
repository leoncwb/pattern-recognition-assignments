import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Carregar datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Adicionar coluna Survived no test (temporária) para concatenação
test_df['Survived'] = -1
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Feature Engineering
combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
combined_df['Title'] = combined_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
combined_df['Title'] = combined_df['Title'].replace(['Mme', 'Lady', 'Countess'], 'Mrs')
combined_df['Title'] = combined_df['Title'].replace(['Dr', 'Rev', 'Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Rare')

combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch']

# Tratar valores ausentes
combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())
combined_df['Embarked'] = combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0])

# Codificar variáveis categóricas
le = LabelEncoder()
combined_df['Sex'] = le.fit_transform(combined_df['Sex'])
combined_df['Embarked'] = le.fit_transform(combined_df['Embarked'])
combined_df['Title'] = le.fit_transform(combined_df['Title'])

# Selecionar features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'Title']
train_clean = combined_df[combined_df['Survived'] != -1]
test_clean = combined_df[combined_df['Survived'] == -1]

X_train = train_clean[features]
y_train = train_clean['Survived']
X_test = test_clean[features]

# Treinar modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"Acurácia média (cross-validation): {scores.mean():.4f}")
rf.fit(X_train, y_train)

# Prever e salvar
y_pred = rf.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test_clean['PassengerId'].astype(int),
    'Survived': y_pred.astype(int)
})
submission.to_csv("submission_assignment2.csv", index=False)
print("Submissão salva em: submission_assignment2.csv")