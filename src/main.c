#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "data.h"
#include "network.h"
#include "utils.h"
#include <time.h>
#include <time.h>

#define TRAINING_SET_SIZE 48000
#define VALIDATION_SET_SIZE 12000
#define TEST_SET_SIZE 10000


int main() {
    clock_t t_start, t_end;
    t_start = clock();

    // Prepare data
    Dataset dataset = prepare_data();

    // Initialize NN
    int sizes[] = {784, 256, 128, 10};
    Network net = initialize_network(sizes, 4);

    // Set parameters
    const int number_of_epochs = 13;
    const int batch_size = 64;

    // Train the network
    train_network(&net, &dataset, number_of_epochs, batch_size);

    // Final evaluation on testing data
    double test_acc = evaluate_accuracy(&net, dataset.test_data, dataset.test_labels);
    printf("Test accuracy: %.2f%%\n", test_acc * 100.0);

    // Export predictions
    clock_t export_start, export_end;
    export_start = clock();
    export_predictions(&net, dataset.train_data, "train_predictions.csv", false);
    export_predictions(&net, dataset.validation_data, "train_predictions.csv", true);
    export_predictions(&net, dataset.test_data, "test_predictions.csv", false);
    export_end = clock();
    printf("Export took: %.3f s\n", (double)(export_end - export_start)/CLOCKS_PER_SEC);

    // Free memory
    free_dataset(&dataset);
    free_network(&net);

    t_end = clock();
    printf("Time elapsed: %.3f s\n", (double)(t_end - t_start)/CLOCKS_PER_SEC);
    return 0;
}
