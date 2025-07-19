# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# 1. Chargement et préparation des données
# ==============================
file_path = "Cryotherapy.csv"  # Assure-toi que le fichier est dans le même dossier
data = pd.read_csv(file_path, decimal=',')

X = data.drop('Result_of_Treatment', axis=1)
y = data['Result_of_Treatment']

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# PCA pour visualisation en 2D
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k')
plt.xlabel('X0')
plt.ylabel('X1')
plt.title('Visualisation des données en 2D (PCA)')
plt.show()

# ==============================
# 2. Modèle linéaire (SGD) pour comparaison
# ==============================
clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
sgd_acc = clf.score(X_test, y_test)
print(f"Accuracy du modèle linéaire : {sgd_acc:.2f}")

# ==============================
# 3. Réseau de Neurones Profond (DNN)
# ==============================
model = Sequential([
    Dense(16, activation='relu', input_dim=X.shape[1]),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping pour éviter le sur-apprentissage
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_split=0.2,
                    epochs=300, batch_size=8, verbose=0,
                    callbacks=[early_stop])

# Évaluation du DNN
dnn_loss, dnn_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy du réseau de neurones : {dnn_acc:.2f}")


# ==============================
# 4. Courbes d'apprentissage
# ==============================


plt.figure(figsize=(12, 5))

# Courbe Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Courbe de la perte')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Courbe Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Courbe de l’accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================
# 5. Frontière de décision du DNN
# ==============================
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Transformer la grille en espace original
grid = np.c_[xx.ravel(), yy.ravel()]
grid_original = pca.inverse_transform(grid)
Z = model.predict(grid_original)
Z = Z.reshape(xx.shape)

# Tracer la frontière
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.7)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=50)
plt.title('Frontière de décision du Réseau de Neurones')
plt.xlabel('Composante PCA 1')
plt.ylabel('Composante PCA 2')
plt.show()


# Résumé des résultats
# ==============================
print("\nRésumé :")
print(f"Modèle Linéaire (SGD) : {sgd_acc*100:.2f}%")
print(f"Réseau de Neurones : {dnn_acc*100:.2f}%")