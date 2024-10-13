#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Fonction pour initialiser un tableau avec des entiers aléatoires
void initializeArray(int* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;  // Nombres aléatoires entre 0 et 99
    }
}

// Fonction pour effectuer l'addition séquentielle de deux tableaux
void addArrays(int* array1, int* array2, int* resultArray, int size) {
    for (int i = 0; i < size; i++) {
        resultArray[i] = array1[i] + array2[i];
    }
}

// Fonction pour mesurer le temps d'exécution
double measureExecutionTime(void (*func)(int*, int), int* array, int size) {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();  // Début de la mesure
    func(array, size);  // Appel de la fonction
    end = clock();  // Fin de la mesure

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;  // Calcul du temps en secondes
    return cpu_time_used;
}

// Fonction pour mesurer le temps d'exécution de l'addition
double measureAdditionTime(void (*func)(int*, int*, int*, int), int* array1, int* array2, int* resultArray, int size) {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();
    func(array1, array2, resultArray, size);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    return cpu_time_used;
}

// Fonction principale
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size_of_arrays>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);  // Conversion de l'argument en entier

    if (size <= 0) {
        printf("Veuillez entrer un entier positif pour la taille du tableau.\n");
        return 1;
    }

    // Allocation de mémoire pour les tableaux
    int* array1 = (int*)malloc(size * sizeof(int));
    int* array2 = (int*)malloc(size * sizeof(int));
    int* resultArray = (int*)malloc(size * sizeof(int));

    if (array1 == NULL || array2 == NULL || resultArray == NULL) {
        printf("Échec de l'allocation de mémoire !\n");
        return 1;
    }

    // Initialiser le générateur de nombres aléatoires
    srand(time(NULL));

    // Mesure du temps d'exécution pour l'initialisation des tableaux
    double initTime1 = measureExecutionTime(initializeArray, array1, size);
    double initTime2 = measureExecutionTime(initializeArray, array2, size);

    // Mesure du temps d'exécution pour l'addition des tableaux
    double additionTime = measureAdditionTime(addArrays, array1, array2, resultArray, size);

    // Affichage des résultats
    printf("Array 1: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array1[i]);
    }
    printf("\n");

    printf("Array 2: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", array2[i]);
    }
    printf("\n");

    printf("Result Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", resultArray[i]);
    }
    printf("\n");

    // Affichage des temps d'exécution
    printf("Temps d'initialisation du tableau 1: %f secondes\n", initTime1);
    printf("Temps d'initialisation du tableau 2: %f secondes\n", initTime2);
    printf("Temps d'exécution de l'addition: %f secondes\n", additionTime);

    // Libération de la mémoire allouée
    free(array1);
    free(array2);
    free(resultArray);

    return 0;
}


