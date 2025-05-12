#include "brian.h"
#include "math.h"
#include "stdlib.h"
#include "stdarg.h"
#include "stdio.h"

#define LAYER_ACTIVATION(func_name) void layer_ ## func_name (Layer* layer) {\
    for (int i = 0; i < layer->output_count; ++i)\
        layer->activated_neurons[i] = func_name (layer->unactivated_neurons[i]);\
}\
void layer_ ## func_name ## _p (Layer* layer) {\
    for (int i = 0; i < layer->output_count; ++i)\
        layer->primed_neurons[i] = func_name ## _prime (layer->unactivated_neurons[i]);\
}


double tanh_prime(double x) {
    return 1 - tanh(x) * tanh(x); 
}

LAYER_ACTIVATION(tanh);

Activation TANH = {
    .func = layer_tanh,
    .prime = layer_tanh_p
};

double leaky_relu(double x) {
    return (x > 0) * x + 0.1 * x * (x < 0); 
}

double leaky_relu_prime(double x) {
    return (x > 0) * 1 + 0.1 * (x < 0); 
}

LAYER_ACTIVATION(leaky_relu);

Activation LEAKY_RELU = {
    .func = layer_leaky_relu,
    .prime = layer_leaky_relu_p
};

double lin(double x) {
    return x;
}

double lin_prime(double x) {
    return 1;
}

LAYER_ACTIVATION(lin);

Activation LINEAR = {
    .func = layer_lin,
    .prime = layer_lin_p
};

double sigmoid(double x) {
    return exp(x) / (1 + exp(x));
}

double sigmoid_prime(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

LAYER_ACTIVATION(sigmoid);

Activation SIGMOID = {
    .func = layer_sigmoid,
    .prime = layer_sigmoid_p
};

void softmax(Layer* layer) {
    double denominator = 0;
    for (int i = 0; i < layer->output_count; ++i)
        denominator += exp(layer->unactivated_neurons[i]);
    
    for (int i = 0; i < layer->output_count; ++i)
        layer->activated_neurons[i] = exp(layer->unactivated_neurons[i]) / denominator;
}

double* curr_y;
void softmax_p(Layer* layer) {
    for (int i = 0; i < layer->output_count; ++i) {
        layer->primed_neurons[i] = 1; 
    }
}

Activation SOFTMAX = {
    .func = softmax,
    .prime = softmax_p
};

Activation* ACTIVATION_TABLE[512] = {
    &TANH,
    &LEAKY_RELU,
    &SIGMOID,
    &LINEAR,
    &SOFTMAX
};

double cost(double* x, double* y, int count) {
    double loss = 0;
    for (int i = 0; i < count; ++i) {
        double f = (x[i] - y[i]);
        loss += f*f;
    }
    return loss * 0.5;
}

double cost_prime(double x, double y) {
    return x - y;
}

Layer* new_layer(int input_count, int output_count, Activation* activation) {
    int weight_count = input_count * output_count; 
    int bias_count = output_count;

    Layer* layer = malloc(sizeof(Layer));
    layer->input_count = input_count;
    layer->output_count = output_count;

    layer->weights = malloc(sizeof(double) * weight_count);
    layer->biases = malloc(sizeof(double) * bias_count);

    layer->weight_changes = malloc(sizeof(double) * weight_count);
    layer->bias_changes = malloc(sizeof(double) * bias_count);

    layer->total_weight_changes = malloc(sizeof(double) * weight_count);
    layer->total_bias_changes = malloc(sizeof(double) * bias_count);
    
    // initialise weights and biases with random values
    for (int i = 0; i < weight_count; ++i)
        layer->weights[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

    for (int i = 0; i < bias_count; ++i)
        layer->biases[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

    layer->unactivated_neurons = malloc(sizeof(double) * output_count);
    layer->activated_neurons = malloc(sizeof(double) * output_count);
    layer->primed_neurons = malloc(sizeof(double) * output_count);

    layer->activation = activation;
    return layer;
}

NN* new_nn(int layer_count, ...) {
    NN* nn = malloc(sizeof(NN));
    nn->layer_count = layer_count;
    nn->layer_sizes = malloc(sizeof(int) * layer_count);
    nn->layers = malloc(sizeof(Layer*) * (layer_count - 1));
    
    va_list args;
    va_start(args, layer_count);
    
    for (int i = 0; i < layer_count; ++i) {
        printf("Getting vararg\n");
        nn->layer_sizes[i] = va_arg(args, int);
        printf("Got it! %d\n", i);
        if (i > 0) {
            Activation* activation = va_arg(args, Activation*);
            nn->layers[i - 1] = new_layer(nn->layer_sizes[i - 1], nn->layer_sizes[i], activation); 
        }
    }

    va_end(args);
    nn->input_count = nn->layer_sizes[0];
    nn->output_count = nn->layer_sizes[nn->layer_count - 1];

    printf("Done!\n");

    return nn;
}

#define GET_WEIGHT(layer, input_index, output_index) (layer)->weights[output_index * (layer)->input_count + input_index]
void layer_forward(Layer* layer, double* inputs) {
    for (int j = 0; j < layer->output_count; ++j) {
        layer->unactivated_neurons[j] = layer->biases[j];
        for (int i = 0; i < layer->input_count; ++i) {
            layer->unactivated_neurons[j] += inputs[i] * GET_WEIGHT(layer, i, j);
        }
    }
    layer->activation->func(layer);
}

double* nn_forward(NN* nn, double* inputs) {
    layer_forward(nn->layers[0], inputs);
    for (int i = 1; i < nn->layer_count - 1; ++i) {
        layer_forward(nn->layers[i], nn->layers[i - 1]->activated_neurons);
    }

    return nn->layers[nn->layer_count - 2]->activated_neurons;
}

void train(NN *nn, int data_len, int data_instance_len, double* xs, double* ys, int epochs, double rate) {
    int instances_count = data_len / data_instance_len; 
    int i = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        
        if (epoch % 100 == 0) {
            printf("EPOCH %d WITH LOSS %lf\n", epoch, get_loss(nn, xs, ys, data_len));
        }

        // reset total changes
        for (int i = 0; i < nn->layer_count - 1; ++i) {
            Layer* layer = nn->layers[i];
            for (int b = 0; b < layer->output_count; ++b)
                layer->total_bias_changes[b] = 0; 

            for (int w = 0; w < layer->input_count * layer->output_count; ++w)
                layer->total_weight_changes[w] = 0; 
        }
        for (int di = 0; di < data_instance_len; ++di) {
            double* x = xs + data_instance_len * i + di * nn->input_count;
            double* y = ys + data_instance_len * i + di * nn->output_count;            
            
            if ((x + nn->input_count) >= xs + data_len)
                break;

            train_one(nn, x, y);
        }

        // apply changes
        for (int i = 0; i < nn->layer_count - 1; ++i) {
            Layer* layer = nn->layers[i];
            for (int b = 0; b < layer->output_count; ++b) {
                layer->biases[b] -= rate * layer->total_bias_changes[b] / (double)data_instance_len; 
            }

            for (int w = 0; w < layer->input_count * layer->output_count; ++w) {
                layer->weights[w] -= rate * layer->total_weight_changes[w] / (double)data_instance_len; 
            }
        }
            
        i = (i + 1) % instances_count;
    }
}

void train_one(NN* nn, double* x, double* y) {
    curr_y = y;
    nn_forward(nn, x);
    // all the last biases and weights
    Layer* last_layer = nn->layers[nn->layer_count - 2];
    last_layer->activation->prime(last_layer);

    for (int i = 0; i < nn->output_count; ++i) {
        last_layer->bias_changes[i] = cost_prime(last_layer->activated_neurons[i], y[i]) * last_layer->primed_neurons[i]; 
    }

    for (int w = 0; w < last_layer->input_count * last_layer->output_count; ++w) {
        int input_index = w % last_layer->input_count;
        int output_index = w / last_layer->input_count;

        double prev_neuron_value = nn->layers[nn->layer_count - 3]->activated_neurons[input_index];
        last_layer->weight_changes[w] = prev_neuron_value * last_layer->bias_changes[output_index];
    }

    // I w1 b1 w2 b2 C
    for (int i = nn->layer_count - 3; i >= 0; --i) {
        Layer* layer = nn->layers[i];
        layer->activation->prime(layer);

        for (int b = 0; b < layer->output_count; ++b) {
            layer->bias_changes[b] = 0; 
            for (int b_prev = 0; b_prev < nn->layers[i + 1]->output_count; ++b_prev) {
                double db_prev = nn->layers[i + 1]->bias_changes[b_prev];
                layer->bias_changes[b] += db_prev * GET_WEIGHT(nn->layers[i + 1], b, b_prev); 
            }
            layer->bias_changes[b] *= layer->primed_neurons[b]; 
        }
        for (int w = 0; w < layer->input_count * layer->output_count; ++w) {
            int input_index = w % layer->input_count;
            int output_index = w / layer->input_count;

            double prev_neuron_value; 
            if (i == 0)
                prev_neuron_value = x[input_index];
            else
                prev_neuron_value = nn->layers[i - 1]->activated_neurons[input_index];

            layer->weight_changes[w] = prev_neuron_value * layer->bias_changes[output_index];
        }
    }
    
    // update total weight and biases changes
    for (int i = 0; i < nn->layer_count - 1; ++i) {
        Layer* layer = nn->layers[i];
        for (int b = 0; b < layer->output_count; ++b)
            layer->total_bias_changes[b] += layer->bias_changes[b]; 

        for (int w = 0; w < layer->input_count * layer->output_count; ++w)
            layer->total_weight_changes[w] += layer->weight_changes[w];
    }
}

int get_activation_code(Activation* activation) {
    for (int i = 0; i < sizeof(ACTIVATION_TABLE); ++i) {
        if (ACTIVATION_TABLE[i] == activation) return i;
    }
    return -1;
}

void save_nn(NN* nn, FILE* file) {
    // layout: 
    // layer_count: <INT>
    // input_layer_size <INT>
    // layers: <layer_count many>
    // -layer_output_size: <INT>
    // -activation: <INT>
    // -layer_weights: <DOUBLE>
    // -layer_biases: <DOUBLE>
    fwrite(&nn->layer_count, sizeof(int), 1, file);
    fwrite(&nn->input_count, sizeof(int), 1, file);

    for (int i = 0; i < nn->layer_count - 1; ++i) {
        Layer* layer = nn->layers[i];
        fwrite(&layer->output_count, sizeof(int), 1, file);
        int activation_code = get_activation_code(layer->activation);
        fwrite(&activation_code, sizeof(int), 1, file);
        fwrite(layer->weights, sizeof(double), layer->input_count * layer->output_count, file);
        fwrite(layer->biases, sizeof(double), layer->output_count, file);
    }
}

NN* load_nn(FILE* file) {
    NN* nn = malloc(sizeof(NN));

    fread(&nn->layer_count, sizeof(int), 1, file);
    nn->layer_sizes = malloc(sizeof(int) * nn->layer_count);
    nn->layers = malloc(sizeof(Layer*) * (nn->layer_count - 1));

    fread(&nn->input_count, sizeof(int), 1, file);
    nn->layer_sizes[0] = nn->input_count;

    for (int i = 1; i < nn->layer_count; ++i) {
        fread(&nn->layer_sizes[i], sizeof(int), 1, file); 
        int activation_code;
        fread(&activation_code, sizeof(int), 1, file);
        Activation* activation = ACTIVATION_TABLE[activation_code]; 
        nn->layers[i - 1] = new_layer(nn->layer_sizes[i - 1], nn->layer_sizes[i], activation); 

        Layer* layer = nn->layers[i - 1];
        fread(layer->weights, sizeof(double), layer->input_count * layer->output_count, file);
        fread(layer->biases, sizeof(double), layer->output_count, file);
    }

    nn->output_count = nn->layer_sizes[nn->layer_count - 1];

    return nn;
}

double get_loss(NN* nn, double* xs, double* ys, int dataset_size) {
    double loss = 0;
    for (int i = 0; i < dataset_size; ++i) {
        double* inputs = &xs[i * nn->input_count];
        double* output = nn_forward(nn, inputs); 
        double* y = &ys[i * nn->output_count];
        loss += cost(output, y, nn->output_count);
    }
    return loss / dataset_size;
}
