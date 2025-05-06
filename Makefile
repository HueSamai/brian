all: test

test:
	gcc -g brian.c test.c -o test.exe
	test.exe
