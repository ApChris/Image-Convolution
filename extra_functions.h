#ifndef EXTRA_FUNCTIONS_H
#define EXTRA_FUNCTIONS_H

typedef enum {NORTHWEST = 0, NORTH, NORTHEAST, WEST, CENTER,
    EAST, SOUTHWEST, SOUTH, SOUTHEAST} dir;

MPI_Datatype type_column;

MPI_Datatype type_column_rgb;

void column_type(int blockwidth, int blockheight);

void get_neighbours(int cur_rank, MPI_Comm cartesian, int ranks[]);

unsigned int northern_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);
unsigned int center_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);
unsigned int southern_row(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);

unsigned int western_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);
unsigned int center_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);
unsigned int eastern_column(unsigned int offset,int bytes_at_offset,int block_img_width, int block_img_height, unsigned char *filter, unsigned char *initial_buffer);

#endif