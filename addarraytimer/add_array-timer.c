#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Fonction pour effectuer l'addition séquentielle des tableaux
void addArrays(int *array1, int *array2, int *resultArray, int size)
{
    for (int i = 0; i < size; i++)
    {
        resultArray[i] = array1[i] + array2[i]; // Addition élément par élément
    }
}

// Fonction pour initialiser un tableau avec des entiers aléatoires
void initializeArray(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 100; // Nombres aléatoires entre 0 et 99
    }
}

// Fonction pour calculer la moyenne des temps
double mean(double *times, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += times[i];
    }
    return sum / n;
}

// Fonction pour calculer l'écart-type des temps
double stddev(double *times, int n, double mean)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += (times[i] - mean) * (times[i] - mean);
    }
    return sqrt(sum / n);
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <number_of_executions>\n", argv[0]);
        return 1;
    }

    int numExecutions = atoi(argv[1]); // Nombre d'exécutions pour chaque taille

    if (numExecutions < 1)
    {
        printf("Veuillez entrer un entier positif pour le nombre d'exécutions.\n");
        return 1;
    }

    // Définir la plage des tailles de tableau selon une échelle logarithmique (base 2)
    int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < numSizes; s++)
    {
        int size = sizes[s];
        printf("Taille du tableau : %d\n", size);

        // Allocation de mémoire pour les tableaux sur l'hôte (CPU)
        int *array1 = (int *)malloc(size * sizeof(int));
        int *array2 = (int *)malloc(size * sizeof(int));
        int *resultArray = (int *)malloc(size * sizeof(int));

        if (array1 == NULL || array2 == NULL || resultArray == NULL)
        {
            printf("Échec de l'allocation de mémoire !\n");
            return 1;
        }

        // Initialiser le générateur de nombres aléatoires
        srand(time(NULL));
        initializeArray(array1, size);
        initializeArray(array2, size);

        // Variables pour stocker les temps d'exécution
        double *executionTimes = (double *)malloc(numExecutions * sizeof(double));

        // Exécuter la fonction d'addition plusieurs fois et mesurer les temps
        for (int i = 0; i < numExecutions; i++)
        {
            clock_t start = clock();

            // Addition des tableaux
            addArrays(array1, array2, resultArray, size);

            clock_t end = clock();
            executionTimes[i] = ((double)(end - start)) / CLOCKS_PER_SEC; // Temps en secondes
        }

        // Calcul de la moyenne et de l'écart-type des temps d'exécution
        double meanTime = mean(executionTimes, numExecutions);
        double stddevTime = stddev(executionTimes, numExecutions, meanTime);

        // Affichage des résultats
        printf("Moyenne du temps d'exécution: %f secondes\n", meanTime);
        printf("Écart-type du temps d'exécution: %f secondes\n", stddevTime);

        // Libérer la mémoire allouée
        free(array1);
        free(array2);
        free(resultArray);
        free(executionTimes);

        printf("\n");
    }

    return 0;
}