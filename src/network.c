#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include "utils.h"
#include "time.h"


void initialize_layer(Layer *layer, int input_size, int number_of_neurons) {
    layer->input_size = input_size;
    layer->number_of_neurons = number_of_neurons;

    int w_count = input_size * number_of_neurons;

    layer->weights = malloc(w_count * sizeof(double));
    layer->biases = malloc(number_of_neurons * sizeof(double));

    layer->inner_potential = malloc(number_of_neurons * sizeof(double));
    layer->output = malloc(number_of_neurons * sizeof(double));
    layer->delta = malloc(number_of_neurons * sizeof(double));

    layer->gradient_weights = malloc(w_count * sizeof(double));
    layer->gradient_biases = malloc(number_of_neurons * sizeof(double));

    // Initialize weights
    double scale = sqrt(2.0 / input_size);  // Normal He initialization
    for (int j = 0; j < number_of_neurons; j++) {
        for (int i = 0; i < input_size; i++) {
            layer->weights[j * input_size + i] = sample_normal() * scale;
        }
        layer->biases[j] = 0.0;
    }
}

Network initialize_network(int *sizes, int num_layers) {
    Network net;
    net.num_layers = num_layers - 1;  // without input             
    net.layers = malloc(net.num_layers * sizeof(Layer));

    for (int l = 0; l < net.num_layers; l++) {
        initialize_layer(&net.layers[l], sizes[l], sizes[l+1]);
    }

    return net;
}

void free_layer(Layer *layer) {
    free(layer->weights);
    free(layer->biases);
    free(layer->inner_potential);
    free(layer->output);
    free(layer->delta);
    free(layer->gradient_weights);
    free(layer->gradient_biases);
}

void free_network(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        free_layer(&net->layers[l]);
    }
    free(net->layers);
}


void forward_pass(Network *net, double *input_vector) {
    double *output_prev = input_vector;

    for (int l = 0; l < net->num_layers; l++) {
        Layer *current_layer = &net->layers[l];
        
        // compute inner potential of each neuron
        for (int j = 0; j < current_layer->number_of_neurons; j++) {
            double sum = current_layer->biases[j];

            // pointer to the beginning of weights to neuron j
            double *weights_to_j = &current_layer->weights[j * current_layer->input_size];
            for (int i = 0; i < current_layer->input_size; i++) {
                sum += weights_to_j[i] * output_prev[i];
            }

            current_layer->inner_potential[j] = sum;
        }

        // apply activation function
        if (l == net->num_layers - 1) {
            // Output layer
            softmax(current_layer->inner_potential, current_layer->output, current_layer->number_of_neurons);
        } else {
            // Hidden layers
            for (int j = 0; j < current_layer->number_of_neurons; j++) {
                current_layer->output[j] = relu(current_layer->inner_potential[j]);
            }
        }

        // output of this layer is the input for the next layer
        output_prev = current_layer->output;
    }
}

// compute the vector of expected outputs of individual neurons
void expected_outputs(int label, double *expected_outputs, int size) {
    for (int i = 0; i < size; i++) {
        expected_outputs[i] = 0.0;
    }
    expected_outputs[label] = 1.0;
}

void backpropagation(Network *net, int label, double *input_vector) {
    int number_of_layers = net->num_layers;

    // last layer
    Layer *last_layer = &net->layers[number_of_layers - 1]; 
    double exp_outputs[10]; // 10 neurons
    expected_outputs(label, exp_outputs, last_layer->number_of_neurons);
    for (int i = 0; i < last_layer->number_of_neurons; i++) {
        last_layer->delta[i] = last_layer->output[i] - exp_outputs[i];
    }

    // hidden layers
    for (int l = number_of_layers - 2; l >= 0; l--) {
        Layer *current_layer = &net->layers[l];
        Layer *layer_above = &net->layers[l + 1];

        for (int i = 0; i < current_layer->number_of_neurons; i++) {
            double sum = 0.0;
            for (int j = 0; j < layer_above->number_of_neurons; j++) {
                double w_ji = layer_above->weights[j * layer_above->input_size + i];
                sum += layer_above->delta[j] * w_ji;
            }
            current_layer->delta[i] = sum * relu_derivative(current_layer->inner_potential[i]); 
        }
    }

    // compute the gradient, accumulate over batch examples
    double *output_prev = input_vector;
    for (int l = 0; l < net->num_layers; l++) {
        Layer *current_layer = &net->layers[l];

        for (int j = 0; j < current_layer->number_of_neurons; j++) {
            double delta_j = current_layer->delta[j];

            double *weights_to_j = &current_layer->gradient_weights[j * current_layer->input_size];
            for (int i = 0; i < current_layer->input_size; i++) {
                weights_to_j[i] += delta_j * output_prev[i];
            }

            current_layer->gradient_biases[j] += delta_j;
        }

        output_prev = net->layers[l].output;
    }
}

// reset gradients to zero
void zero_gradients(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];

        int w_count = layer->input_size * layer->number_of_neurons;
        for (int i = 0; i < w_count; i++) {
            layer->gradient_weights[i] = 0.0;
        }

        for (int j = 0; j < layer->number_of_neurons; j++) {
            layer->gradient_biases[j] = 0.0;
        }  
    }
}

void update_weights(Network *net, double learning_rate, int batch_size) {
    for (int l = 0; l < net->num_layers; l++) {
        Layer *layer = &net->layers[l];

        int w_count = layer->input_size * layer->number_of_neurons;
        for (int i = 0; i < w_count; i++) {
            layer->weights[i] -= learning_rate * (layer->gradient_weights[i] / batch_size);
        }

        for (int j = 0; j < layer->number_of_neurons; j++) {
            layer->biases[j] -= learning_rate * (layer->gradient_biases[j] / batch_size);
        }
    }
}

int predict(Network *net, double *input) {
    forward_pass(net, input);
    Layer *output_layer = &net->layers[net->num_layers - 1];
    int best = 0;
    for (int i = 1; i < output_layer->number_of_neurons; i++) {
        if (output_layer->output[i] > output_layer->output[best]) {
            best = i;
        }
    }
    return best;
}

void shuffle_data(Vectors *vectors, Labels *labels) {
    for (int i = vectors->number_of_vectors - 1; i > 0; i--) {
        int j = rand() % (i + 1);

        // swap vectors
        for (int k = 0; k < vectors->vector_size; k++) {
            float tmp = vectors->data[i][k];
            vectors->data[i][k] = vectors->data[j][k];
            vectors->data[j][k] = tmp;
        }

        // swap labels
        int tmp_label = labels->data[i];
        labels->data[i] = labels->data[j];
        labels->data[j] = tmp_label;
    }
}

// TRAINING
int train_network(Network *net, Dataset *dataset, int number_of_epochs, int batch_size) {

    double learning_rate = 0.1;
    for (int epoch = 0; epoch < number_of_epochs; epoch++) {
        if (epoch < 5){
            learning_rate = 0.1;
        }
        else if (epoch < 10){
            learning_rate = 0.05;
        }
        else{
            learning_rate = 0.01;
        }

        shuffle_data(&dataset->train_data, &dataset->train_labels);

        double epoch_loss = 0;
        int epoch_correct = 0;
        int epoch_samples = 0;

        double t_train_start = clock();
        printf("Epoch: %d\n", epoch);

        // for every batch...
        for (int i = 0; i < TRAINING_SET_SIZE; i += batch_size) {  
            zero_gradients(net);

            int end = i + batch_size;
            if (end > TRAINING_SET_SIZE) {
                end = TRAINING_SET_SIZE;
            }

            // for every vector/image...
            for (int k = i; k < end; k++) {
                double *vector = dataset->train_data.data[k];
                int label = dataset->train_labels.data[k];                

                forward_pass(net, vector);

                // cross-entropy loss and accuracy
                double p = net->layers[net->num_layers-1].output[label];
                epoch_loss += -log(p);
                if (predict(net, vector) == label) {
                    epoch_correct++;
                }

                backpropagation(net, label, vector);
            }

            update_weights(net, learning_rate, end - i);

            epoch_samples += (end - i);
        }

        double t_train_end = clock();
        printf("Training of the epoch took %.3f s\n", (double)(t_train_end - t_train_start)/CLOCKS_PER_SEC);
        printf("Training data accuracy: %.2f%%\n", 100.0 * (double)epoch_correct / epoch_samples);
        printf("Validation data accuracy:  %.2f%%\n", 100.0 * evaluate_accuracy(net, dataset->validation_data, dataset->validation_labels));
    }

    return 0;
}