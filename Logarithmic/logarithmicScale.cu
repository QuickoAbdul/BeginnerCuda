#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

// Kernel CUDA pour l'addition des tableaux sur le GPU
__global__ void addArrays(int *array1, int *array2, int *resultArray, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calcul de l'indice global pour chaque thread
    if (i < size)
    {
        resultArray[i] = array1[i] + array2[i]; // Addition élément par élément
    }
}

// Fonction pour initialiser un tableau avec des entiers aléatoires sur l'hôte (CPU)
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
        int *h_array1 = (int *)malloc(size * sizeof(int));
        int *h_array2 = (int *)malloc(size * sizeof(int));
        int *h_resultArray = (int *)malloc(size * sizeof(int));

        if (h_array1 == NULL || h_array2 == NULL || h_resultArray == NULL)
        {
            printf("Échec de l'allocation de mémoire !\n");
            return 1;
        }

        // Initialiser le générateur de nombres aléatoires
        srand(time(NULL));
        initializeArray(h_array1, size);
        initializeArray(h_array2, size);

        // Allocation de mémoire sur le GPU
        int *d_array1;
        int *d_array2;
        int *d_resultArray;
        cudaMalloc((void **)&d_array1, size * sizeof(int));
        cudaMalloc((void **)&d_array2, size * sizeof(int));
        cudaMalloc((void **)&d_resultArray, size * sizeof(int));

        // Copier les tableaux de l'hôte vers le périphérique
        cudaMemcpy(d_array1, h_array1, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array2, h_array2, size * sizeof(int), cudaMemcpyHostToDevice);

        // Variables pour stocker les temps d'exécution
        double *executionTimes = (double *)malloc(numExecutions * sizeof(double));

        // Exécuter le kernel plusieurs fois et mesurer les temps
        for (int i = 0; i < numExecutions; i++)
        {
            cudaEvent_t start, stop;
            float milliseconds = 0;

            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            int threadsPerBlock = 256;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

            cudaEventRecord(start);
            addArrays<<<1, 1>>>(d_array1, d_array2, d_resultArray, size);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);

            executionTimes[i] = milliseconds / 1000.0; // Convertir en secondes
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        // Calcul de la moyenne et de l'écart-type des temps d'exécution
        double meanTime = mean(executionTimes, numExecutions);
        double stddevTime = stddev(executionTimes, numExecutions, meanTime);

        printf("Moyenne du temps d'exécution: %f secondes\n", meanTime);
        printf("Écart-type du temps d'exécution: %f secondes\n", stddevTime);

        // Libérer la mémoire allouée
        free(h_array1);
        free(h_array2);
        free(h_resultArray);
        free(executionTimes);

        cudaFree(d_array1);
        cudaFree(d_array2);
        cudaFree(d_resultArray);

        printf("\n");
    }

    return 0;
}
