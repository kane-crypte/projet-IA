# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 12:46:04 2025

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
import random
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pickle


# === 2. Chargement et prétraitement des données ===

# Charger les données
df = pd.read_excel("Cryotherapy.xlsx")
print("Colonnes :", df.columns.tolist())

# Encodage one-hot (pour 'sex' et 'Type')
df_encoded = pd.get_dummies(df, columns=['sex', 'Type'], drop_first=True)

# Séparation features / target
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

# Conversion float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")


test_data = X_test.copy()
test_data["Result_of_Treatment"] = y_test.values

# Enregistrer dans un fichier Excel
test_data.to_excel("donnees_test.xlsx", index=False)
print("Fichier 'donnees_test.xlsx' sauvegardé avec succès.")

# Combine les features d'entraînement avec la cible
train_data = X_train.copy()
train_data["Result_of_Treatment"] = y_train.values

# Enregistrer dans un fichier Excel
train_data.to_excel("donnees_train.xlsx", index=False)
print(" Fichier 'donnees_train.xlsx' sauvegardé avec succès.")


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

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# === 5. Évaluation ===

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrécision sur les données de test : {accuracy:.2f}")

# Prédictions
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy (sklearn):", accuracy_score(y_test, y_pred))

# === 6. Affichage des courbes ===
# === 6 bis. Visualisation 2D de la séparation des classes ===

# Appliquer PCA sur toutes les features normalisées
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Création du DataFrame PCA
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Result_of_Treatment'] = y.values

# Affichage
plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df['PC1'], 
    pca_df['PC2'], 
    c=pca_df['Result_of_Treatment'], 
    cmap='coolwarm', edgecolors='k', s=80
)
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('Projection PCA : Visualisation des classes')
plt.grid(True)
plt.show()


# Loss et Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Courbe de loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Courbe d\'accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion")
plt.show()

# Comparaison accuracy
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
test_acc = accuracy

labels = ['Train Accuracy', 'Test Accuracy']
values = [train_acc * 100, test_acc * 100]

plt.bar(labels, values, color=['blue', 'green'])
plt.ylim([0, 100])
plt.ylabel('Accuracy (%)')
plt.title('Performance du modèle : Train vs Test')
for i, v in enumerate(values):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center')
plt.show()

# === 7. Interface graphique ===

from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Charger le modèle et le scaler
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = float(data['age'])
    time = float(data['time'])
    number = float(data['number'])
    area = float(data['area'])
    sex_male = 1 if data['sex'] == 'male' else 0
    type2 = 1 if data['type'] == 'Type 2' else 0
    type3 = 1 if data['type'] == 'Type 3' else 0

    # Créer un DataFrame
    input_data = pd.DataFrame([{
        "age": age, "Time": time, "Number_of_Warts": number,
        "Total_Area": area, "sex_male": sex_male,
        "Type_Type 2": type2, "Type_Type 3": type3
    }])

    # Normaliser
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]

    result = "Success" if prediction > 0.5 else "Failure"
    return jsonify({"prediction": result, "probability": float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
