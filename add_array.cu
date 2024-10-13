#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Kernel CUDA pour l'addition des tableaux sur le GPU
__global__ void addArrays(int* array1, int* array2, int* resultArray, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calcul de l'indice global pour chaque thread
    if (i < size) {
        resultArray[i] = array1[i] + array2[i];  // Addition élément par élément
    }
}

// Fonction pour initialiser un tableau avec des entiers aléatoires sur l'hôte (CPU)
void initializeArray(int* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;  // Nombres aléatoires entre 0 et 99
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size_of_arrays>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);  // Conversion de l'argument de la ligne de commande en entier

    if (size <= 0) {
        printf("Veuillez entrer un entier positif pour la taille du tableau.\n");
        return 1;
    }

    // Allocation de mémoire pour les tableaux sur l'hôte (CPU)
    int* h_array1 = (int*)malloc(size * sizeof(int));
    int* h_array2 = (int*)malloc(size * sizeof(int));
    int* h_resultArray = (int*)malloc(size * sizeof(int));

    if (h_array1 == NULL || h_array2 == NULL || h_resultArray == NULL) {
        printf("Échec de l'allocation de mémoire !\n");
        return 1;
    }

    // Initialiser le générateur de nombres aléatoires
    srand(time(NULL));

    // Initialisation des tableaux sur l'hôte (CPU)
    initializeArray(h_array1, size);
    initializeArray(h_array2, size);

    // Allocation de mémoire pour les tableaux sur le périphérique (GPU)
    int* d_array1;
    int* d_array2;
    int* d_resultArray;
    
    cudaMalloc((void**)&d_array1, size * sizeof(int));
    cudaMalloc((void**)&d_array2, size * sizeof(int));
    cudaMalloc((void**)&d_resultArray, size * sizeof(int));

    // Copier les tableaux de l'hôte vers le périphérique
    cudaMemcpy(d_array1, h_array1, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, h_array2, size * sizeof(int), cudaMemcpyHostToDevice);

    // Définir la taille des blocs et des grilles
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Lancer le kernel d'addition sur le GPU
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_array1, d_array2, d_resultArray, size);

    // Copier le résultat du périphérique vers l'hôte
    cudaMemcpy(h_resultArray, d_resultArray, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Afficher les tableaux et le résultat
    printf("Array 1: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array1[i]);
    }
    printf("\n");

    printf("Array 2: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_array2[i]);
    }
    printf("\n");

    printf("Result Array: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", h_resultArray[i]);
    }
    printf("\n");

    // Libérer la mémoire allouée sur l'hôte et le périphérique
    free(h_array1);
    free(h_array2);
    free(h_resultArray);

    cudaFree(d_array1);
    cudaFree(d_array2);
    cudaFree(d_resultArray);

    return 0;
}

