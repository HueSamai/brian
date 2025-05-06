#ifndef BRIAN_H
#define BRIAN_H

typedef struct {
    double (*func)(double);
    double (*prime)(double);
} Activation;

extern Activation TANH;
extern Activation LEAKY_RELU;
extern Activation LINEAR; 

typedef struct {
    int input_count;
    int output_count;
    double* weights;
    double* biases;
    Activation* activation;

    // private
    double* unactivated_neurons;
    double* activated_neurons;

    double* weight_changes;
    double* bias_changes;

    double* total_weight_changes;
    double* total_bias_changes;
} Layer;

typedef struct {
    int layer_count;
    int* layer_sizes;
    Layer** layers; 
    int input_count;
    int output_count;
} NN;

double cost(double* x, double* y, int count);
double cost_prime(double* x, double* y, int count);

Layer* new_layer(int input_count, int output_count, Activation*);
NN* new_nn(int layer_count, ...);                            // layer sizes follow

void train(NN* nn, int data_len, int data_instance_len, double* xs, double* ys, int epochs, double rate);

void train_one(NN* nn, double* xs, double* ys);

void layer_forward(Layer* layer, double* inputs); 
double* nn_forward(NN* nn, double* inputs);

#endif
