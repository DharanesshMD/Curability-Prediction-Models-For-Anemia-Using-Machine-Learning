import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
import pandas as pd

# Load model and preprocessors
model_bundle = joblib.load('anemia_best_model.joblib')
model = model_bundle['model']
gender_encoder = model_bundle['gender_encoder']
imputer = model_bundle['imputer']
scaler = model_bundle['scaler']
selector = model_bundle['selector']
selected_features = model_bundle['selected_features']

# Load dataset
df = pd.read_csv('synthetic_anemia_data.csv')
X_all = df.drop('Anemia', axis=1).copy()
X_all['Gender'] = gender_encoder.transform(X_all['Gender'])
X_all_imp = imputer.transform(X_all)
X_all_scl = scaler.transform(X_all_imp)
X_all_sel = selector.transform(X_all_scl)

# Predict for all
all_preds = model.predict(X_all_sel)
all_probs = model.predict_proba(X_all_sel)[:, 1]

def show_results():
    anemia_count = np.sum(all_preds == 1)
    no_anemia_count = np.sum(all_preds == 0)
    total = len(all_preds)
    anemia_pct = anemia_count / total * 100
    no_anemia_pct = no_anemia_count / total * 100
    prob_min = np.min(all_probs)
    prob_max = np.max(all_probs)
    prob_mean = np.mean(all_probs)
    message = (
        f'Total people: {total}\n'
        f'Predicted Anemia: {anemia_count} ({anemia_pct:.1f}%)\n'
        f'Predicted No Anemia: {no_anemia_count} ({no_anemia_pct:.1f}%)\n'
        f'Probability of Anemia:\n'
        f'  Min: {prob_min:.2%}\n'
        f'  Max: {prob_max:.2%}\n'
        f'  Mean: {prob_mean:.2%}'
    )
    messagebox.showinfo('Dataset Prediction Summary', message)

root = tk.Tk()
root.title('Anemia Curability Prediction - Dataset Summary')

mainframe = ttk.Frame(root, padding='20')
mainframe.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

summary_label = ttk.Label(mainframe, text='Click the button to see anemia prediction summary for the dataset:')
summary_label.grid(row=0, column=0, pady=10)

predict_btn = ttk.Button(mainframe, text='Show Results', command=show_results)
predict_btn.grid(row=1, column=0, pady=10)

root.mainloop() 