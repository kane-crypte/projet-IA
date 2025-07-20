# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 15:22:26 2025

@author: lass
"""

# === 1. Importation des bibliothèques ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tkinter as tk
from tkinter import messagebox
import numpy as np

# === 2. Chargement et prétraitement des données ===
df = pd.read_excel("Cryotherapy.xlsx")

# Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# Encodage one-hot
df_encoded = pd.get_dummies(df, columns=['sex', 'Type'], drop_first=True)

# Features et target
X = df_encoded.drop("Result_of_Treatment", axis=1)
y = df_encoded["Result_of_Treatment"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Conversion en float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# === 3. Création du modèle MLP ===
model = Sequential([
    Dense(16, input_dim=X_train.shape[1], activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(8, activation='relu',
          kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# === 4. Entraînement ===
model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === 5. Interface graphique ===
BG_COLOR = "#2C3E50"
FG_COLOR = "#ECF0F1"
BUTTON_COLOR = "#3498DB"
BUTTON_HOVER = "#2980B9"
ENTRY_BG = "#34495E"

def on_enter(e):
    e.widget['background'] = BUTTON_HOVER

def on_leave(e):
    e.widget['background'] = BUTTON_COLOR


def import_test_data():
    """Charge une ligne aléatoire du fichier Excel (données brutes) dans l'UI."""
    random_index = np.random.randint(0, len(df))
    test_sample = df.iloc[random_index]

    entry_age.delete(0, tk.END)
    entry_age.insert(0, test_sample["age"])

    entry_time.delete(0, tk.END)
    entry_time.insert(0, test_sample["Time"])

    entry_number.delete(0, tk.END)
    entry_number.insert(0, test_sample["Number_of_Warts"])

    if "Area" in test_sample:
        entry_area.delete(0, tk.END)
        entry_area.insert(0, test_sample["Area"])
    else:
        messagebox.showwarning("Avertissement", "La colonne 'Area' est absente dans le fichier Excel.")

    # Gestion du sexe : 1 = homme, 2 = femme
    if test_sample["sex"] == 1:
        var_sex.set("male")
    else:
        var_sex.set("female")

    # Gestion du type : 1, 2, 3
    var_type.set(f"Type {int(test_sample['Type'])}")




def make_prediction():
    try:
        # Conversion sécurisée en float
        age = float(entry_age.get().strip())
        time = float(entry_time.get().strip())
        number = float(entry_number.get().strip())
        area = float(entry_area.get().strip())
        sex = var_sex.get()
        wart_type = var_type.get()

        # Construire les features
        input_data = {
            "age": [age],
            "Time": [time],
            "Number_of_Warts": [number],
            "Area": [area],
            "sex_male": [1 if sex == "male" else 0],
            "Type_Type 2": [1 if wart_type == "Type 2" else 0],
            "Type_Type 3": [1 if wart_type == "Type 3" else 0],
        }

        input_df = pd.DataFrame(input_data)

        # Ajouter colonnes manquantes
        for col in X_train.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[X_train.columns]

        # Force toutes les colonnes en float32
        input_df = input_df.astype("float32")

        # Normalisation
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        prediction = model.predict(input_df_scaled)[0][0]
        result = "✅ Traitement efficace" if prediction > 0.5 else "❌ Échec du traitement"
        messagebox.showinfo("Résultat de la prédiction", f"{result}\n(Probabilité : {prediction:.2f})")

    except ValueError as ve:
        messagebox.showerror("Erreur", f"Valeur invalide : {ve}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))


# ==== Création interface ====
window = tk.Tk()
window.title("Cryotherapy Prediction")
window.geometry("400x550")
window.config(bg=BG_COLOR)

title = tk.Label(window, text="Cryotherapy Prediction",
                 font=("Segoe UI", 16, "bold"), fg=FG_COLOR, bg=BG_COLOR)
title.pack(pady=10)

def add_label_entry(text):
    tk.Label(window, text=text, fg=FG_COLOR, bg=BG_COLOR,
             font=("Segoe UI", 11, "bold")).pack(pady=(8,0))
    entry = tk.Entry(window, font=("Segoe UI", 11),
                     bg=ENTRY_BG, fg=FG_COLOR, insertbackground='white', relief="flat")
    entry.pack(pady=4, ipady=5, ipadx=5)
    return entry

entry_age = add_label_entry("Âge :")
entry_time = add_label_entry("Durée (Time) :")
entry_number = add_label_entry("Nombre de verrues :")
entry_area = add_label_entry("Zone totale :")

tk.Label(window, text="Sexe :", fg=FG_COLOR, bg=BG_COLOR,
         font=("Segoe UI", 11, "bold")).pack(pady=5)
var_sex = tk.StringVar(value="male")
tk.Radiobutton(window, text="Homme", variable=var_sex, value="male",
               bg=BG_COLOR, fg=FG_COLOR, selectcolor=ENTRY_BG).pack()
tk.Radiobutton(window, text="Femme", variable=var_sex, value="female",
               bg=BG_COLOR, fg=FG_COLOR, selectcolor=ENTRY_BG).pack()

tk.Label(window, text="Type de verrue :", fg=FG_COLOR, bg=BG_COLOR,
         font=("Segoe UI", 11, "bold")).pack(pady=5)
var_type = tk.StringVar(value="Type 2")
menu = tk.OptionMenu(window, var_type, "Type 1", "Type 2", "Type 3")
menu.config(bg=ENTRY_BG, fg=FG_COLOR, font=("Segoe UI", 10), relief="flat")
menu.pack(pady=5)

def create_button(text, command, color):
    btn = tk.Button(window, text=text, command=command, bg=color,
                    fg="white", font=("Segoe UI", 11, "bold"), relief="flat",
                    activebackground=BUTTON_HOVER, activeforeground="white")
    btn.pack(pady=8, ipadx=5, ipady=5, fill="x", padx=40)
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn

create_button("Importer données ", import_test_data, BUTTON_COLOR)
create_button("Prédire le résultat", make_prediction, "#27AE60")

window.mainloop()
