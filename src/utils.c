#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "data.h"
#include "network.h"

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

void softmax(double *inner_potential, double *output, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(inner_potential[i]);  
        sum += output[i];
    }
        
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}


// [0,1] normal distribution https://en.wikipedia.org/wiki/Marsaglia_polar_method
double sample_normal(void) {
    double u, v, s;
    do {
        u = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s == 0.0 || s >= 1.0);

    s = sqrt(-2.0 * log(s) / s);
    return u * s;
}

double evaluate_accuracy(Network *net, Vectors test_data, Labels test_labels) {
    int correct = 0;

    for (int i = 0; i < test_data.number_of_vectors; i++) {
        int prediction = predict(net, test_data.data[i]);
        if (prediction == test_labels.data[i]) {
            correct++;
        }  
    }

    return (double)correct / test_data.number_of_vectors;
}

void export_predictions(Network *net, Vectors vectors, const char *filename, bool append) {
    FILE *file = fopen(filename, append ? "a" : "w");
    if (!file) {
        printf("Error: Cannot open %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < vectors.number_of_vectors; i++) {
        int prediction = predict(net, vectors.data[i]);
        fprintf(file, "%d\n", prediction);
    }

    fclose(file);
}
