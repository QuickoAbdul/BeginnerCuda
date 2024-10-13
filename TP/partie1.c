#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Prototypes des fonctions
void initializeArrays(int *A, int *B, int size);
void addArrays(int *A, int *B, int *C, int size);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <size_of_arrays>\n", argv[0]);
        return 1; // Sortie avec une erreur si la taille n'est pas fournie
    }

    int size = atoi(argv[1]); // Convertir l'argument en entier

    // Allouer de la mémoire pour les tableaux
    int *A = (int *)malloc(size * sizeof(int));
    int *B = (int *)malloc(size * sizeof(int));
    int *C = (int *)malloc(size * sizeof(int));

    // Initialiser les tableaux
    initializeArrays(A, B, size);

    // Additionner les tableaux
    addArrays(A, B, C, size);

    // Afficher les résultats
    printf("Tableau A: ");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");

    printf("Tableau B: ");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", B[i]);
    }
    printf("\n");

    printf("Tableau C (A + B): ");
    for (int i = 0; i < size; i++)
    {
        printf("%d ", C[i]);
    }
    printf("\n");

    // Libération de la mémoire
    free(A);
    free(B);
    free(C);

    return 0;
}

// Implémentation des fonctions
void initializeArrays(int *A, int *B, int size)
{
    srand(time(NULL)); // Initialiser le générateur de nombres aléatoires

    for (int i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // Remplir A avec des valeurs aléatoires (0 à 99)
        B[i] = rand() % 100; // Remplir B avec des valeurs aléatoires (0 à 99)
    }
}

void addArrays(int *A, int *B, int *C, int size)
{
    for (int i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i]; // Additionner les éléments correspondants
    }
}
