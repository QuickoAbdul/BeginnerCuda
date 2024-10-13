#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
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

// Fonction pour obtenir le temps actuel
double getCurrentTime()
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER currentTime;
    QueryPerformanceFrequency(&frequency);                    // Récupérer la fréquence
    QueryPerformanceCounter(&currentTime);                    // Récupérer le compteur
    return (double)currentTime.QuadPart / frequency.QuadPart; // Temps en secondes
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
        printf("Veuillez entrer un entier positif pour le nombre d'executions.\n");
        return 1;
    }

    // Initialiser le générateur de nombres aléatoires
    srand(time(NULL));

    // Définir la plage des tailles de tableau
    int sizes[] = {16, 32, 64, 128, 256, 512, 1024, 2024};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    // Boucle à travers les tailles de tableau
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
            printf("Echec de l'allocation de memoire !\n");
            return 1;
        }

        initializeArray(array1, size);
        initializeArray(array2, size);

        // Variables pour stocker les temps d'exécution
        double *executionTimes = (double *)malloc(numExecutions * sizeof(double));

        // Exécuter la fonction d'addition plusieurs fois et mesurer les temps
        for (int exec = 0; exec < numExecutions; exec++)
        {
            double start = getCurrentTime();

            // Addition des tableaux
            addArrays(array1, array2, resultArray, size);

            double end = getCurrentTime();
            executionTimes[exec] = end - start; // Temps en secondes
            printf("Execution %d - Temps d'execution : %.10f secondes\n", exec + 1, executionTimes[exec]);
        }

        // Calcul de la moyenne et de l'écart-type des temps d'exécution
        double meanTime = mean(executionTimes, numExecutions);
        double stddevTime = stddev(executionTimes, numExecutions, meanTime);

        // Affichage des résultats
        printf("Moyenne du temps d'execution : %.10f secondes\n", meanTime);
        printf("Ecart-type du temps d'execution : %.10f secondes\n", stddevTime);

        // Libérer la mémoire allouée
        free(array1);
        free(array2);
        free(resultArray);
        free(executionTimes);

        printf("\n");
    }

    return 0;
}
