#include "brian.h"
#include "stdio.h"
#include "raylib.h"
#include "stdlib.h"
#include "time.h"

#define DATASET_SIZE 60000
double xs[DATASET_SIZE * 784];
int labels[DATASET_SIZE];
double ys[DATASET_SIZE * 10];
#define BATCH_SIZE 60000

NN* nn = NULL; 
FILE* logfile;

double _get_loss() {
    double loss = 0;
    for (int i = 0; i < DATASET_SIZE; ++i) {
        double* inputs = &xs[i * nn->input_count];
        double* output = nn_forward(nn, inputs); 
        double* y = &ys[i * nn->output_count];
        loss += cost(output, y, nn->output_count);
    }
    return loss;
}

void print_loss() {
    double loss = _get_loss();
    fprintf(logfile, "Total loss: %lf\n", loss);
}

void read_in_data() {
    FILE* image_file, *label_file;

    // images
    fopen_s(&image_file, "mnist/train-images.idx3-ubyte", "rb");
    unsigned char c;
    
    fgetc(image_file);
    fgetc(image_file);
    fgetc(image_file);
    int h = fgetc(image_file);
    
    for (int j = 0; j < h * 4; ++j)
        fgetc(image_file);

    int i = 0;
    while (i < DATASET_SIZE * 784) {
        c = fgetc(image_file);
        xs[i++] = c / 255.0;
    }

    fclose(image_file);

    fopen_s(&label_file, "mnist/train-labels.idx1-ubyte", "rb");
    
    fgetc(label_file);
    fgetc(label_file);
    fgetc(label_file);
    h = fgetc(label_file);
    
    for (int j = 0; j < h * 4; ++j)
        fgetc(label_file);

    i = 0;
    while (i < DATASET_SIZE) {
        c = fgetc(label_file);
        labels[i] = c;
        for (int n = 0; n < 10; ++n) {
            ys[i * 10 + n] = labels[i] == n;
        }
        ++i;
    }
    fclose(label_file);
}

double rate = 0.05;
double loss;
void mnist_train() {
    print_loss();
    train(nn, DATASET_SIZE, BATCH_SIZE, xs, ys, 100 * (DATASET_SIZE / BATCH_SIZE), rate);
    print_loss();
    loss = _get_loss();
}

void mnist_test() {
    time_t t;
    time(&t);
    srand(t);
    int index = rand() % DATASET_SIZE;

    if (nn == NULL) {
        FILE* nn_file;
        if (fopen_s(&nn_file, "mnist.nn", "rb")) {
            nn = new_nn(3, 
                784,
                34, &LEAKY_RELU,
                10, &SOFTMAX
            );
        } else {
            nn = load_nn(nn_file);
            fclose(nn_file);
        }
    }
    loss = _get_loss();

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            index = rand() % DATASET_SIZE;
        }
        if (IsKeyPressed(KEY_I)) {
            rate *= 2;
        }
        BeginDrawing();
            ClearBackground(BLACK);
            if (IsKeyPressed(KEY_T)) {
                DrawText("Training!", 0, 0, 64, RAYWHITE);
                EndDrawing();
                mnist_train();
                index = rand() % DATASET_SIZE;
            }
            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    char v = xs[index * 784 + y * 28 + x] * 255;
                    Color col = { v, v, v, 255 };
                    DrawRectangle(x * 20, y * 20, 20, 20, col);
                }
            }
            char buffer[256];
            sprintf(buffer, "Number: %d", labels[index]);
            DrawText(buffer, 0, 0, 32, WHITE);
            sprintf(buffer, "Rate: %lf", rate);
            DrawText(buffer, 400, 0, 20, WHITE);
            sprintf(buffer, "Loss: %lf", loss);
            DrawText(buffer, 400, 20, 20, WHITE);
            double* guess = nn_forward(nn, &xs[index * 784]);
            for (int i = 0; i < 10; ++i) {
                sprintf(buffer, "%d: %lf", i, guess[i]);
                DrawText(buffer, 0,  32 + i * 20, 20, WHITE);
            }
        EndDrawing();
    }

    FILE* nn_file;
    fopen_s(&nn_file, "mnist.nn", "wb");
    save_nn(nn, nn_file);
    fclose(nn_file);
}

int main() {
    fopen_s(&logfile, "log", "w");
    read_in_data();
    InitWindow(280*2, 280*2, "MNIST");
    mnist_test();
    fclose(logfile);
    return 0;
}
