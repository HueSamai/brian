#include "stdio.h"
#include "brian.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"

void print_loss(NN* nn, int data_len, double* xs, double* ys) {
    double loss = 0;
    for (int i = 0; i < data_len; ++i) {
        double* inputs = &xs[i * nn->input_count];
        double* output = nn_forward(nn, inputs); 
        double* y = &ys[i * nn->output_count];
        loss += cost(output, y, nn->output_count);
    }

    printf("Total loss: %lf\n", loss);
}

#define PI 3.1415926535
int main_sin() {
    time_t t;
    time(&t);
    srand((double)t);
    
    NN* nn = new_nn(4, 
        1, 
        10, &LEAKY_RELU,
        10, &LEAKY_RELU,
        1, &TANH
    );
    printf("Created nn!\n");

    int data_len = 99;
    double xs[data_len];
    double ys[data_len];

    int j = 0;
    for (double i = 0; i < 2 * PI; i += 2 * PI / data_len) {
        xs[j] = i; 
        ys[j] = sin(i); 
        ++j;
    }

    for (int i = 0; i < data_len; ++i) {
        double* inputs = &xs[i];
        double* output = nn_forward(nn, inputs); 
        printf("Input: %lf Output: %lf\n", inputs[0], output[0]);
    }
    print_loss(nn, data_len, xs, ys);
    printf("Training!\n");
    train(nn, data_len, data_len, xs, ys, 100000, 5e-1);

    for (int i = 0; i < data_len; ++i) {
        double* inputs = xs + i;
        double* output = nn_forward(nn, inputs); 
        printf("Input: %lf Output: %lf\n", inputs[0], output[0]);
        
    }
    print_loss(nn, data_len, xs, ys);

    return 0;
}

int main_xor() {
    time_t t;
    time(&t);
    srand((double)t);
    NN* nn = new_nn(3, 
        2,
        10, &LEAKY_RELU,
        1, &LEAKY_RELU 
    );

    double xs[] = {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    };

    double ys[] = {
        0,
        1,
        1,
        0, 
    };

    for (int i = 0; i < 4; ++i) {
        double* inputs = xs + i * 2;
        double* output = nn_forward(nn, inputs); 
        printf("Input: %lf %lf Output: %lf\n", inputs[0], inputs[1], output[0]);
        
    }
    print_loss(nn, 4, xs, ys);
    printf("Training!\n");
    train(nn, 4, 4, xs, ys, 999999, 2e-4);
    for (int i = 0; i < 4; ++i) {
        double* inputs = xs + i * 2;
        double* output = nn_forward(nn, inputs); 
        printf("Input: %lf %lf Output: %lf\n", inputs[0], inputs[1], output[0]);
        
    }
    print_loss(nn, 4, xs, ys);


    return 0;
}

int main_prob() {
    time_t t;
    time(&t);
    srand((double)t);

    NN* nn = new_nn(3, 
        2,
        10, &LEAKY_RELU,
        3, &SOFTMAX
    );

    int data_len = 100;
    double xs[data_len * 2];
    double ys[data_len * 3];

    for (int i = 0; i < data_len; ++i) {
        xs[i * 2] = (double)rand() / (double)RAND_MAX;
        xs[i * 2 + 1] = (double)rand() / (double)RAND_MAX;
        ys[i * 3] = (double)(xs[i] > xs[i + 1]);
        ys[i * 3 + 1] = (double)(xs[i] < xs[i + 1]);
        ys[i * 3 + 2] = (double)(xs[i] == xs[i + 1]);
    }
    print_loss(nn, data_len, xs, ys);
    printf("Training!\n");
    train(nn, data_len, data_len, xs, ys, 99999, 5e-1);
    print_loss(nn, data_len, xs, ys);

    return 0;
}

int main() {
    return main_prob();
    return main_sin();
    //return main_xor();
    return 0;
}
