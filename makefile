
all: mpi_conv omp_conv

mpi_conv: main.o convolution.o extra_functions.o
	mpicc -o mpi_conv main.o convolution.o extra_functions.o -std=c99 -lm -Wall -Werror

omp_conv: main.o convolution_omp.o extra_functions.o
	mpicc -fopenmp -o omp_conv main.o convolution.o extra_functions.o -std=c99 -lm -Wall -Werror

main.o: main.c
	mpicc -c -lm -std=c99 -Wall -Werror -o main.o main.c

convolution.o: convolution.c
	mpicc -c -lm -std=c99 -Wall -Werror -o convolution.o convolution.c

convolution_omp.o: convolution.c
	mpicc -c -lm -std=c99 -Wall -Werror -fopenmp -o convolution_omp.o convolution.c

extra_functions.o: extra_functions.c
	mpicc -c -lm -std=c99 -Wall -Werror -o extra_functions.o extra_functions.c

clean:
	rm -f mpi_conv omp_conv main.o convolution.o convolution_omp.o

rebuild: clean all
