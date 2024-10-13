#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Fonction pour effectuer l'addition des tableaux sur le GPU
__global__ void addArraysCUDA(int *array1, int *array2, int *resultArray, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Index global
    if (i < size)
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

// Fonction pour initialiser un tableau avec des 1
void initializeArrayWithOnes(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = 1; // Remplir avec des 1
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
    if (argc != 4)
    {
        printf("Usage: %s <num_executions> <threads_per_block> <blocks_per_grid>\n", argv[0]);
        return 1;
    }

    int numExecutions = atoi(argv[1]); // Utiliser le premier argument pour le nombre d'exécutions
    int threadsPerBlock = atoi(argv[2]);
    int blocksPerGrid = atoi(argv[3]);

    if (numExecutions < 1)
    {
        printf("Veuillez entrer un entier positif pour le nombre d'executions.\n");
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
            printf("Echec de l'allocation de memoire !\n");
            return 1;
        }

        // Initialiser le générateur de nombres aléatoires
        srand(time(NULL));
        initializeArrayWithOnes(array1, size);
        initializeArrayWithOnes(array2, size);

        // Allocation de mémoire sur le GPU
        int *d_array1, *d_array2, *d_resultArray;
        cudaMalloc((void **)&d_array1, size * sizeof(int));
        cudaMalloc((void **)&d_array2, size * sizeof(int));
        cudaMalloc((void **)&d_resultArray, size * sizeof(int));

        // Copier les données de l'hôte vers le GPU
        cudaMemcpy(d_array1, array1, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array2, array2, size * sizeof(int), cudaMemcpyHostToDevice);

        // Pour mesurer les temps d'exécution
        double *execTimes = (double *)malloc(numExecutions * sizeof(double));

        // Boucle pour exécuter le calcul plusieurs fois
        for (int exec = 0; exec < numExecutions; exec++)
        {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start); // Démarrer le chronomètre
            addArraysCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_array1, d_array2, d_resultArray, size);
            cudaDeviceSynchronize(); // Attendre la fin de l'exécution du GPU
            cudaEventRecord(stop);   // Arrêter le chronomètre

            // Calculer le temps d'exécution
            cudaEventSynchronize(stop); // Assurer que l'événement d'arrêt est terminé
            float execTime;
            cudaEventElapsedTime(&execTime, start, stop); // Temps en millisecondes
            execTimes[exec] = execTime / 1000.0;          // Convertir en secondes
            printf("Execution %d - Temps d'execution (CUDA) : %.10f secondes\n", exec + 1, execTimes[exec]);

            // Copier le résultat du GPU vers l'hôte
            cudaMemcpy(resultArray, d_resultArray, size * sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Calculer et afficher la moyenne et l'écart-type des temps d'exécution
        double avgTime = mean(execTimes, numExecutions);
        double stddevTime = stddev(execTimes, numExecutions, avgTime);
        printf("Temps moyen : %.10f secondes, Ecart-type : %.10f secondes\n", avgTime, stddevTime);

        // Libérer la mémoire GPU et hôte
        cudaFree(d_array1);
        cudaFree(d_array2);
        cudaFree(d_resultArray);
        free(array1);
        free(array2);
        free(resultArray);
        free(execTimes);

        printf("\n");
    }

    return 0;
}
