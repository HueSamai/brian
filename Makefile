RAYLIB_DIR =C:\raylib\raylib\src

all: test
	
test:
	gcc -g brian.c test.c -o test.exe
	test.exe

mnist:
	gcc brian.c mnist.c -o mnist.exe -O3 -I$(RAYLIB_DIR) -L$(RAYLIB_DIR) -lraylib -lopengl32 -lgdi32 -lwinmm -mwindows
	mnist

imgnn:
	gcc -g brian.c imgnn.c -o imgnn.exe -O3 -I$(RAYLIB_DIR) -L$(RAYLIB_DIR) -lraylib -lopengl32 -lgdi32 -lwinmm -mwindows
	imgnn test_img.jpeg 
