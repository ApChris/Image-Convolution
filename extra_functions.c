
#include "convolution.h"
#include "extra_functions.h"

unsigned int northern_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return (initial_buffer[offset-block_img_width-bytes_at_offset] * filter[SOUTHEAST] + initial_buffer[offset-block_img_width] * filter[SOUTH] + initial_buffer[offset-block_img_width+bytes_at_offset] * filter[SOUTHWEST] );
}

unsigned int center_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return (initial_buffer[offset] * filter[CENTER] + initial_buffer[offset-bytes_at_offset] * filter[EAST] + initial_buffer[offset+bytes_at_offset] * filter[WEST]);
}
unsigned int southern_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return (initial_buffer[offset+block_img_width-bytes_at_offset] * filter[NORTHEAST] + initial_buffer[offset+block_img_width] * filter[NORTH] + initial_buffer[offset+block_img_width+bytes_at_offset] * filter[NORTHWEST] );
}

unsigned int western_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return (initial_buffer[offset-block_img_width-bytes_at_offset] * filter[SOUTHEAST] + initial_buffer[offset-1] * filter[EAST] + initial_buffer[offset+block_img_width-bytes_at_offset] * filter[NORTHEAST] );
}
unsigned int center_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return ( initial_buffer[offset] * filter[CENTER] + initial_buffer[offset-block_img_width] * filter[SOUTH] + initial_buffer[offset+block_img_width] * filter[NORTH] );
}
unsigned int eastern_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer)
{
    return ( initial_buffer[offset-block_img_width+bytes_at_offset] * filter[SOUTHWEST] + initial_buffer[offset+bytes_at_offset] * filter[WEST] + initial_buffer[offset+block_img_width+bytes_at_offset] * filter[NORTHWEST] );
}


// prepare the column type
void column_type(int block_img_width, int block_img_height) 
{
    // Creates a vector (strided) datatype
    MPI_Type_vector(block_img_height, 1, block_img_width, MPI_CHAR,
    &type_column);
    // Commits the datatype
    MPI_Type_commit(&type_column);

    MPI_Type_vector(block_img_height, 3, block_img_width, MPI_CHAR,
    &type_column_rgb);
    MPI_Type_commit(&type_column_rgb);
}

//  Get neighbours for each calling process(rank)

void get_neighbours(int current_rank, MPI_Comm cart, int array_ranks[]) 
{
    int array_coordinations[2];

    MPI_Cart_coords(cart, current_rank, 2, array_coordinations);
    
    array_coordinations[0]--;
    array_coordinations[1]--;
    
   // printf("rank=%d, [0]=%d, [1]= %d\n",current_rank,array_coordinations[1],array_coordinations[0] );

// FOR EXAMPLE
//    -1   0  1
// -1  nw  N  NE
//  0  W   C  E
//  1  sw  D  SE

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[NORTHWEST]);
    array_coordinations[1]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[NORTH]);
    array_coordinations[1]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[NORTHEAST]);
    array_coordinations[0]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[EAST]);
    array_coordinations[1] -= 2;

    array_ranks[CENTER] = current_rank;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[WEST]);
    array_coordinations[0]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[SOUTHWEST]);
    array_coordinations[1]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[SOUTH]);
    array_coordinations[1]++;

    MPI_Cart_rank(cart, array_coordinations, &array_ranks[SOUTHEAST]);
}
