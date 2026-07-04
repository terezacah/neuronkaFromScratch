#ifndef DATA_H
#define DATA_H

#define IMAGE_SIZE 784  // 28x28

// Structs
typedef struct {
    int number_of_vectors;
    int vector_size;
    double (*data)[IMAGE_SIZE];  // can be accessed as data[row][col]
} Vectors;

typedef struct {
    int size;
    int *data;
} Labels;

typedef struct {
    Vectors train_data;
    Labels train_labels;
    Vectors validation_data;
    Labels validation_labels;
    Vectors test_data;
    Labels test_labels;
} Dataset;

// Functions
Vectors load_vectors_from_csv(const char *path, int rows, int cols, int start_row);
Labels load_labels_from_csv(const char *path, int rows, int start_row);
Dataset prepare_data();
void free_dataset(Dataset *dataset);

#endif