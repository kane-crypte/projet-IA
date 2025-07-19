#include <stdio.h>
#define N 21       // Nombre d'exemples d'apprentissage
#define DIM 3      // Nombre de caractéristiques (biais + 2 valeurs)
#define MAX_EPOCHS 1

// Fonction d'activation (step)
int activation(double somme) {
    if (somme > 0)
        return 1;
    else
        return 0;
}

// Produit scalaire entre deux vecteurs
double produit_scalaire(double v1[], double v2[]) {
    double total = 0;
    for (int i = 0; i < DIM; i++) {
        total += v1[i] * v2[i];
    }
    return total;
}

// Fonction d'apprentissage du perceptron
void train(double X[][DIM], int Y[], double w[], int max_epochs) {
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        int erreurs = 0;
        printf("\nEpoch %d\n", epoch + 1);
        printf("Entrées\t\tSortie Réelle\tPrédiction\tErreur\t\tPoids (w)\n");
        
        for (int i = 0; i < N; i++) {
            int y_pred = activation(produit_scalaire(X[i], w));
            int erreur = Y[i] - y_pred;
            
            printf("[%.1f %.1f %.1f]\t%d\t\t%d\t\t%d\t\t[%.2f %.2f %.2f]\n",
                   X[i][0], X[i][1], X[i][2], Y[i], y_pred, erreur, w[0], w[1], w[2]);
            
            if (erreur != 0) {
                erreurs++;
                for (int j = 0; j < DIM; j++) {
                    w[j] += erreur * X[i][j];
                }
            }
        }
        printf("Total erreurs cette epoch : %d\n", erreurs);
    }
}

// Prédiction simple
int predict(double x[], double w[]) {
    return activation(produit_scalaire(x, w));
}

int main() {
    // Données d'apprentissage (biais + longueur sépale + longueur pétale)
    double X[N][DIM] = {
        {1, 5.1, 1.4}, {1, 5.2, 3.9}, {1, 4.9, 1.4}, {1, 6.6, 4.6}, {1, 4.7, 1.3},
        {1, 6.6, 4.6}, {1, 4.6, 1.5}, {1, 4.9, 3.3}, {1, 5.0, 1.4}, {1, 6.3, 4.7},
        {1, 5.4, 1.7}, {1, 5.7, 4.5}, {1, 4.6, 1.4}, {1, 6.5, 4.6}, {1, 5.0, 1.5},
        {1, 5.5, 4.0}, {1, 4.4, 1.4}, {1, 6.9, 4.9}, {1, 4.9, 1.5}, {1, 7.0, 4.7},
        {1, 6.4, 4.5}
    };
    int Y[N] = {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1};

    // Poids initiaux = 1
    double w[DIM] = {1, 1, 1};

    // Entraînement
    train(X, Y, w, MAX_EPOCHS);
    printf("\nPoids finaux : [%.2f %.2f %.2f]\n", w[0], w[1], w[2]);

    // Données de test
    double test_data[10][DIM] = {
        {1, 5.4, 1.5}, {1, 4.8, 1.6}, {1, 4.8, 1.4}, {1, 4.3, 1.1}, {1, 5.8, 1.2},
        {1, 5.0, 3.5}, {1, 5.9, 4.2}, {1, 6.0, 4.0}, {1, 6.1, 4.7}, {1, 5.6, 3.6}
    };
    int test_labels[10] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    // Test
    printf("\nRésultats du test :\n");
    for (int i = 0; i < 10; i++) {
        int y_pred = predict(test_data[i], w);
        const char *pred_label = (y_pred == 0) ? "Iris-setosa" : "Iris-versicolor";
        const char *real_label = (test_labels[i] == 0) ? "Iris-setosa" : "Iris-versicolor";
        printf("Test %d : Prédit = %s, Réel = %s, Correct = %s\n",
               i + 1, pred_label, real_label, (y_pred == test_labels[i]) ? "Oui" : "Non");
    }

    return 0;
}
