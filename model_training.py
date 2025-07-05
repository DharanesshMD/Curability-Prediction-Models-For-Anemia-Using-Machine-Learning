import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load data
df = pd.read_csv('synthetic_anemia_data.csv')

# Separate features and target
X = df.drop('Anemia', axis=1)
y = df['Anemia']

# Encode categorical variables
gender_encoder = LabelEncoder()
X['Gender'] = gender_encoder.fit_transform(X['Gender'])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# Feature selection (select top 5 features)
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X_scaled.columns[selector.get_support()]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Define models and hyperparameters
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}
params = {
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5, None]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
}

best_score = 0
best_model = None
best_name = ''

for name, model in models.items():
    grid = GridSearchCV(model, params[name], cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name} accuracy: {acc:.4f}')
    if acc > best_score:
        best_score = acc
        best_model = grid.best_estimator_
        best_name = name

print(f'Best model: {best_name} with accuracy {best_score:.4f}')

# Save the best model and preprocessing objects
joblib.dump({
    'model': best_model,
    'gender_encoder': gender_encoder,
    'imputer': imputer,
    'scaler': scaler,
    'selector': selector,
    'selected_features': selected_features.tolist()
}, 'anemia_best_model.joblib')
print('Best model and preprocessors saved to anemia_best_model.joblib') 