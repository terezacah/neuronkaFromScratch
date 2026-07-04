#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"

#define TRAINING_SET_SIZE 48000
#define VALIDATION_SET_SIZE 12000
#define TEST_SET_SIZE 10000

// Free memory of the dataset
void free_dataset(Dataset *dataset) {
    free(dataset->train_data.data);
    free(dataset->train_labels.data);
    free(dataset->validation_data.data);
    free(dataset->validation_labels.data);
    free(dataset->test_data.data);
    free(dataset->test_labels.data);
}

// Load images
Vectors load_vectors_from_csv(const char *path, int rows, int cols, int start_row) {
    Vectors vectors;
    vectors.number_of_vectors = rows;
    vectors.vector_size = cols;
    vectors.data = malloc(rows * sizeof(double[IMAGE_SIZE]));
    if (!vectors.data) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        printf("Error: Cannot open %s\n", path);
        exit(1);
    }

    char buffer[4096];
    for (int row = 0; row < start_row + rows; row++) {
        if (!fgets(buffer, 4096, file)) {
            printf("Error when fetching data\n");
            fclose(file);
            exit(1);
        }
        if (row - start_row >= 0){
            char* token = strtok(buffer, ",");
            for (int col = 0; col < cols; col++) {
                if (!token) {
                    printf("Error when parsing data\n");
                    fclose(file);
                    exit(1);
                }

                int pixel = strtol(token, NULL, 10);
                vectors.data[row - start_row][col] = pixel / 255.0;    // normalize to [0, 1]

                token = strtok(NULL, ",");
            }
        }
    }

    fclose(file);
    return vectors;
}

// Load labels
Labels load_labels_from_csv(const char *path, int rows, int start_row) {
    Labels labels;
    labels.size = rows;
    labels.data = malloc(rows * sizeof(int));
    if (!labels.data) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    FILE *file = fopen(path, "r");
    if (!file) {
        printf("Error: Cannot open %s\n", path);
        exit(1);
    }

    char buffer[128];
    for (int row = 0; row < start_row + rows; row++) {
        if (!fgets(buffer, sizeof(buffer), file)) {
            printf("Error when parsing data labels\n");
            fclose(file);
            exit(1);
        }
        if (row - start_row >= 0){
            labels.data[row - start_row] = strtol(buffer, NULL, 10);
        }
    }

    fclose(file);
    return labels;
}


Dataset prepare_data(){
    Dataset dataset;
    dataset.train_data = load_vectors_from_csv("data/fashion_mnist_train_vectors.csv", TRAINING_SET_SIZE, IMAGE_SIZE, 0);
    dataset.train_labels = load_labels_from_csv("data/fashion_mnist_train_labels.csv", TRAINING_SET_SIZE, 0);

    dataset.validation_data = load_vectors_from_csv("data/fashion_mnist_train_vectors.csv", VALIDATION_SET_SIZE, IMAGE_SIZE, TRAINING_SET_SIZE);
    dataset.validation_labels = load_labels_from_csv("data/fashion_mnist_train_labels.csv", VALIDATION_SET_SIZE, TRAINING_SET_SIZE);

    dataset.test_data = load_vectors_from_csv("data/fashion_mnist_test_vectors.csv", TEST_SET_SIZE, IMAGE_SIZE, 0);
    dataset.test_labels = load_labels_from_csv("data/fashion_mnist_test_labels.csv", TEST_SET_SIZE, 0);

    return dataset;
}
