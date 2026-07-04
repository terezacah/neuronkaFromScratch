#ifndef NETWORK_H
#define NETWORK_H

#include "data.h"

#define TRAINING_SET_SIZE 48000
#define VALIDATION_SET_SIZE 12000
#define TEST_SET_SIZE 10000


// Structs
typedef struct {
    int input_size;             // number of inputs per neuron
    int number_of_neurons;      // number of neurons in the layer

    double *weights;            // [number_of_neurons * input_size]
    double *biases;             // [number_of_neurons]

    double *inner_potential;    // [number_of_neurons]
    double *output;             // [number_of_neurons]

    double *delta;              // ​∂E_k/∂y_j​ * σ_j′​(ξ_j​)
    double *gradient_weights;   // accumulated gradients ∂E_k/​​∂w_ji
    double *gradient_biases;    // accumulated gradients ∂E_k/​​∂b_j
} Layer;

typedef struct {
    int num_layers;             // number of hidden layers + output layer 
    Layer *layers;
} Network;

// Functions 
Network initialize_network(int *sizes, int num_layers);
void initialize_layer(Layer *layer, int input_size, int number_of_neurons);
void free_network(Network *net);

void forward_pass(Network *net, double *input_vector);
void backpropagation(Network *net, int label, double *input_vector);
void zero_gradients(Network *net);
void update_weights(Network *net, double learning_rate, int batch_size);
int predict(Network *net, double *input);
double evaluate_accuracy(Network *net, Vectors test_data, Labels test_labels);
int train_network(Network *net, Dataset *dataset, int epochs, int batch_size);

#endif