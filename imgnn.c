#include "raylib.h"
#include "brian.h"
#include "string.h"
#include <stdio.h>
#include "stdlib.h"
#include "time.h"

int training = 0;
Image image;
Texture2D texture;
NN* nn;
double* xs;
double* ys;
int width, height;

void get_image_from_nn() {
    UnloadImage(image);
    image = GenImageColor(width, height, RAYWHITE);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double inputs[] = {
                x / (double)(width),
                y / (double)(height)
            };
            double* rgb = nn_forward(nn, inputs);
            ImageDrawPixel(&image, x, y, (Color){
                    rgb[0] * 255,
                    rgb[1] * 255,
                    rgb[2] * 255,
                    255,
            });
        }
    }
    
    UnloadTexture(texture);
    texture = LoadTextureFromImage(image);
}

Image training_image;
Texture training_texture;
void get_image_from_training() {
    UnloadImage(training_image);
    training_image = GenImageColor(width, height, RAYWHITE);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double inputs[] = {
                x / (double)(width),
                y / (double)(height)
            };
            double* rgb = &ys[(y * width + x) * 3]; 
            ImageDrawPixel(&training_image, x, y, (Color){ (int)(rgb[0] * 255), rgb[1] * 255, rgb[2] * 255, 255 });
        }
    }
    
    UnloadTexture(training_texture);
    training_texture = LoadTextureFromImage(training_image);
}

double rate = 0.001;
int main(int argc, char** argv) {
    int CANVAS_SIZE = 500;

    char* img_path = argv[1];
    char* img_extension = img_path;
    while (*img_extension != '.' && *img_extension != 0) {
        img_extension++;
    }
    if (*img_extension == '.')
        img_extension++;

    if (!strcmp(img_extension, "imgnn")) {
        // handle displaying image
        FILE* file;
        fopen_s(&file, img_path, "rb");
        nn = load_nn(file); 
        fread(&width, sizeof(int), 1, file);
        fread(&height, sizeof(int), 1, file);
        fclose(file);
        TraceLog(LOG_INFO, "Loaded in nn!\n");
        
        InitWindow(CANVAS_SIZE, CANVAS_SIZE, "imgnn");
        get_image_from_nn();
    } else {
        time_t t;
        time(&t);
        srand(t);

        // handle training nn img
        image = LoadImage(img_path);
        ImageResize(&image, 8, 8);
        nn = new_nn(4,
            2,
            64, &LEAKY_RELU,
            64, &LEAKY_RELU,
            3, &SIGMOID
        );

        width = image.width;
        height = image.height;
        
        xs = malloc(sizeof(double) * 2 * image.width * image.height);
        ys = malloc(sizeof(double) * 3 * image.width * image.height);
        
        int i = 0;
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                xs[i * 2] = x / (double)(image.width);
                xs[i * 2 + 1] = y / (double)(image.height);

                Color colour = GetImageColor(image, x, y);
                ys[i * 3] = colour.r / (double)255;
                ys[i * 3 + 1] = colour.g / (double)255;
                ys[i * 3 + 2] = colour.b / (double)255;
                ++i;
            }
        }


        training = 1;
        // train(nn, width * height, width * height, xs, ys, 500, rate);
        InitWindow(CANVAS_SIZE * 2, CANVAS_SIZE, "imgnn");
        get_image_from_training();
    }


    double train_timer = 0;
    int train_iterations = 10;
    int i = 0;
    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_I))
            rate += 0.05;
        if (IsKeyPressed(KEY_D))
            rate -= 0.05;
        BeginDrawing();
            if (training && train_timer <= 0) {
                train(nn, width * height, width * height, xs, ys, 100, rate);
                get_image_from_nn();
                train_timer = 0.00001;
                if (i++ >= train_iterations) {
                    i = 0;
                }
            } 
            DrawTexturePro(texture, (Rectangle) {0, 0, width, height}, (Rectangle) { 0, 0, CANVAS_SIZE, CANVAS_SIZE }, (Vector2){0,0}, 0, WHITE); 
            DrawTexturePro(training_texture, (Rectangle) {0, 0, width, height}, (Rectangle) { CANVAS_SIZE, 0, CANVAS_SIZE, CANVAS_SIZE }, (Vector2){0,0}, 0, WHITE); 
            char buffer[256];
            sprintf(buffer, "Loss: %lf", get_loss(nn, xs, ys, width * height));
            DrawText(buffer, 0, 0, 32, WHITE);
        EndDrawing();
        train_timer -= GetFrameTime();
    }

    if (training) {
        FILE* file;
        fopen_s(&file, "test_img.imgnn", "wb");
        save_nn(nn, file);
        fwrite(&width, sizeof(int), 1, file);
        fwrite(&height, sizeof(int), 1, file);
        fclose(file);
    }

    return 0;
}
