#ifndef UTILS_H
#define UTILS_H

#include "data.h"
#include <stdbool.h>
// Structs

// Functions
double relu(double x);
double relu_derivative(double x);
void softmax(double *inner_potential, double *output, int size);
double sample_normal(void);
double evaluate_accuracy(Network *net, Vectors test_data, Labels test_labels);
void export_predictions(Network *net, Vectors test_data, const char *filename, bool append);

#endif
