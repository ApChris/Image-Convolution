#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <mpi.h>
#include <stdlib.h>
#include "extra_functions.h"

unsigned char * convolution_function(int block_img_width, int block_img_height, unsigned char *filter, int ranks[], int reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf);

unsigned char * convolution_function_rgb(int block_img_width, int block_img_height, unsigned char *filter, int ranks[], int reps, MPI_Comm cartesian,
    unsigned char *src_buf, unsigned char *dest_buf);

#endif