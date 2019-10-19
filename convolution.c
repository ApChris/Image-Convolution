#include "convolution.h"
#include "extra_functions.h"
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#define THREADNUM 2
#endif

// processing the image total_reps times from initial_buffer to final_buffer
// lastly, after swaping final_buffer with inital_buffer returns the filtered image(initial_buffer)


unsigned char * convolution_function(int block_img_width, int block_img_height, unsigned char *filter, int array_ranks[], int reps, MPI_Comm cart, unsigned char *initial_buffer, unsigned char *final_buffer) 
{
    
    // Allocating space for neighbouring pixels
    unsigned char * north_row = (unsigned char*)malloc(block_img_width);
    unsigned char * south_row = (unsigned char*)malloc(block_img_width);
    unsigned char * west_column = (unsigned char*)malloc(block_img_height);
    unsigned char * east_column = (unsigned char*)malloc(block_img_height);

    // northwest_corner -> north west corner, se -> southeast corner etc...
    unsigned char northwest_corner, northeast_corner, southwest_corner, southeast_corner; 

    //  9 = 4 sides + 4 corners + center
    MPI_Request send_request[9], receive_request[9];

    // sum of all filters
    unsigned int final_filter = 0;
    for (int i = 0; i < 9; i++)
    {
        final_filter += filter[i];
    }

    // For every rep
    for (int i = 0; i < reps; i++) 
    {
        // send pixels to other ranks, 4 corners , 4 sides       
        // sending one element AND MPI_CHAR we get a corner & row. Sending one element AND type_column we get a column(side)
        // intial_buffer[0] = northwest , north , west
        MPI_Isend(&initial_buffer[0], 1, MPI_CHAR, array_ranks[NORTHWEST], NORTHWEST, cart, &send_request[NORTHWEST]);     
        MPI_Isend(&initial_buffer[0], block_img_width, MPI_CHAR, array_ranks[NORTH], NORTH, cart, &send_request[NORTH]);
        MPI_Isend(&initial_buffer[0], 1, type_column, array_ranks[WEST], WEST, cart, &send_request[WEST]);

        // initial_buffer[block_img_width-1]= northeast , east
        MPI_Isend(&initial_buffer[block_img_width-1], 1, MPI_CHAR, array_ranks[NORTHEAST], NORTHEAST, cart, &send_request[NORTHEAST]);
        MPI_Isend(&initial_buffer[block_img_width-1], 1, type_column, array_ranks[EAST], EAST, cart, &send_request[EAST]);

        // initial_buffer[(block_img_height-1)*block_img_width] = southwest , south
        MPI_Isend(&initial_buffer[(block_img_height-1)*block_img_width], 1, MPI_CHAR, array_ranks[SOUTHWEST], SOUTHWEST, cart, &send_request[SOUTHWEST]);
        MPI_Isend(&initial_buffer[(block_img_height-1)*block_img_width], block_img_width, MPI_CHAR, array_ranks[SOUTH], SOUTH, cart, &send_request[SOUTH]);

        // initial_buffer[block_img_height*block_img_width-1] = southeast
        MPI_Isend(&initial_buffer[block_img_height*block_img_width-1], 1, MPI_CHAR, array_ranks[SOUTHEAST], SOUTHEAST, cart, &send_request[SOUTHEAST]);

        // center is calculated independently
        send_request[CENTER] = MPI_REQUEST_NULL;
	
        // get pixels from other ranks, 4 corner, 4 sides
        // northwest , north , west
        MPI_Irecv(&northwest_corner, 1, MPI_CHAR, array_ranks[NORTHWEST], SOUTHEAST, cart, &receive_request[NORTHWEST]);
        MPI_Irecv(north_row, block_img_width, MPI_CHAR, array_ranks[NORTH], SOUTH, cart, &receive_request[NORTH]);
        MPI_Irecv(west_column, block_img_height, MPI_CHAR, array_ranks[WEST], EAST, cart, &receive_request[WEST]);

        // northeast , east
        MPI_Irecv(&northeast_corner, 1, MPI_CHAR, array_ranks[NORTHEAST], SOUTHWEST, cart, &receive_request[NORTHEAST]);       
        MPI_Irecv(east_column, block_img_height, MPI_CHAR, array_ranks[EAST], WEST, cart, &receive_request[EAST]);

        // southwest , south
        MPI_Irecv(&southwest_corner, 1, MPI_CHAR, array_ranks[SOUTHWEST], NORTHEAST, cart, &receive_request[SOUTHWEST]);
        MPI_Irecv(south_row, block_img_width, MPI_CHAR, array_ranks[SOUTH], NORTH, cart, &receive_request[SOUTH]);

        // southeast
        MPI_Irecv(&southeast_corner, 1, MPI_CHAR, array_ranks[SOUTHEAST], NORTHWEST, cart, &receive_request[SOUTHEAST]);

        // center is calculated independently
        receive_request[CENTER] = MPI_REQUEST_NULL;

        // calculate all inside pixels using OPENMP
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(THREADNUM)
        #endif

        for (int i = 1; i < block_img_height-1; i++)
        {
            for (int j = 1; j < block_img_width-1; j++) 
            {
                // find offset of current pixel in array
                // for example width= 25 , height=50 =1250pixels , -> last pixel 49*25 + 25 = 1250

                unsigned int offset_current_pixel = i*block_img_width + j;

                // for inside pixels, just sum all 9 pixels(the grid 3x3) multiplied by their coefficients

                unsigned int pixels_sum = northern_row(offset_current_pixel,1,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += center_row(offset_current_pixel,1,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += southern_row(offset_current_pixel,1,block_img_width,block_img_height,filter,initial_buffer);;

                // divide with the sum of all filters
                pixels_sum /= final_filter;

                // get 8 righter bits
                final_buffer[offset_current_pixel] = (char)pixels_sum & 0xFF;
            }
        }
        // wait until we receive all needed pixels for pixels on the edges


	
        MPI_Waitall(8, receive_request, MPI_STATUSES_IGNORE);
	
        // calculating pixels('grid-3x3') on northern side
        for (int i = 1; i < block_img_width-1; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 2nd and 3rd row of the current 'grid'

            unsigned int pixels_sum = center_row(i,1,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += southern_row(i,1,block_img_width,block_img_height,filter,initial_buffer);
            
            // calculating pixels received from the above rank ( 1st row of the 'grid')
            // for example : these pixels are the southest pixels of the above rank that got connected using send/receive requests
            pixels_sum += north_row[i-1] * filter[SOUTHEAST];
            pixels_sum += north_row[i] * filter[SOUTH];
            pixels_sum += north_row[i+1] * filter[SOUTHWEST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[i] = (char)pixels_sum & 0xFF;
        }

        // calculating pixels('grid-3x3') on southest side
        unsigned int southest_row = (block_img_height-1)*block_img_width;
        for (int i = 1; i < block_img_width-1; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 1st and 2nd row of the current 'grid'

            unsigned int pixels_sum = center_row(southest_row+i,1,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += northern_row(southest_row+i,1,block_img_width,block_img_height,filter,initial_buffer);

            // calculating pixels received from the below rank ( 3nd row of the 'grid')
            // for example : these pixels are the northern pixels of the below rank that got connected using send/receive requests
            pixels_sum += south_row[i-1] * filter[NORTHEAST];
            pixels_sum += south_row[i] * filter[NORTH];
            pixels_sum += south_row[i+1] * filter[NORTHWEST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[southest_row+i] = (char)pixels_sum & 0xFF;
        }

        // calculating pixels('grid-3x3') on western side
        for (int i = 1; i < block_img_height-1; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 2nd and 3rd column of the current 'grid'

            unsigned int west_side_offset = i*block_img_width;
            unsigned int pixels_sum = center_column(west_side_offset,1,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += eastern_column(west_side_offset,1,block_img_width,block_img_height,filter,initial_buffer);

            // calculating pixels received from the left rank ( 1st column of the 'grid')
            // for example : these pixels are the eastern pixels of the left rank that got connected using send/receive requests

            pixels_sum += west_column[i-1] * filter[SOUTHEAST];
            pixels_sum += west_column[i] * filter[EAST];
            pixels_sum += west_column[i+1] * filter[NORTHEAST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[west_side_offset] = (char)pixels_sum & 0xFF;
        }

        // calculating pixels('grid-3x3') on eastern side
        for (int i = 1; i < block_img_height-1; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 1st and 2nd column of the current 'grid'

            unsigned int east_side_offset = (i+1)*block_img_width - 1;
            unsigned int pixels_sum = center_column(east_side_offset,1,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += western_column(east_side_offset,1,block_img_width,block_img_height,filter,initial_buffer);

            // calculating pixels received from the right rank ( 3rd column of the 'grid')
            // for example : these pixels are the eastern pixels of the right rank that got connected using send/receive requests

            pixels_sum += east_column[i-1] * filter[SOUTHWEST];
            pixels_sum += east_column[i] * filter[WEST];
            pixels_sum += east_column[i+1] * filter[NORTHWEST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[east_side_offset] = (char)pixels_sum & 0xFF;
        }

        unsigned int cursor = 0;
        unsigned int pixels_sum = 0;

        // calculating corner pixels
        // northwest corner pixel
        // pixels_sum of the current rank 
        pixels_sum = initial_buffer[0] * filter[CENTER];     
        pixels_sum += initial_buffer[1] * filter[WEST];      
        pixels_sum += initial_buffer[block_img_width] * filter[NORTH];     
        pixels_sum += initial_buffer[block_img_width+1] * filter[NORTHWEST]; 

        // pixels that have received from other ranks
        pixels_sum += north_row[0] * filter[SOUTH];       
        pixels_sum += north_row[1] * filter[SOUTHWEST];       
        pixels_sum += west_column[0] * filter[EAST];
        pixels_sum += west_column[1] * filter[NORTHEAST];
        pixels_sum += northwest_corner * filter[SOUTHEAST];
       
        //divide with the sum of all filters
        pixels_sum /= final_filter;
        // get 8 righter bits because we need <=255 demical value  
        final_buffer[0] = (char)pixels_sum && 0xFF;
        


        // northeast corner pixel
        // pixels_sum of the current rank 
        cursor = block_img_width-1;
        pixels_sum = initial_buffer[cursor] * filter[CENTER];
        pixels_sum += initial_buffer[cursor-1] * filter[EAST];
        pixels_sum += initial_buffer[cursor+block_img_width] * filter[NORTH];
        pixels_sum += initial_buffer[cursor+block_img_width-1] * filter[NORTHEAST];

        // pixels that have received from other ranks
        pixels_sum += north_row[cursor] * filter[SOUTH];
        pixels_sum += north_row[cursor-1] * filter[SOUTHEAST];
        pixels_sum += east_column[0] * filter[WEST];
        pixels_sum += east_column[1] * filter[NORTHWEST];
        pixels_sum += northeast_corner * filter[SOUTHWEST];

        //divide with the sum of all filters
        pixels_sum /= final_filter;
        // get 8 righter bits because we need <=255 demical value
        final_buffer[cursor] = (char)pixels_sum & 0xFF;
        

        // southwest corner pixel
        // pixels_sum of the current rank 
        cursor = (block_img_height-1)*block_img_width;
        pixels_sum = initial_buffer[cursor] * filter[CENTER];
        pixels_sum += initial_buffer[cursor+1] * filter[WEST];
        pixels_sum += initial_buffer[cursor-block_img_width] * filter[SOUTH];
        pixels_sum += initial_buffer[cursor-block_img_width+1] * filter[SOUTHWEST];

        // pixels that have received from other ranks
        pixels_sum += south_row[0] * filter[NORTH];
        pixels_sum += south_row[1] * filter[NORTHWEST];
        pixels_sum += west_column[block_img_height-1] * filter[EAST];
        pixels_sum += west_column[block_img_height-2] * filter[SOUTHEAST];
        pixels_sum += southwest_corner * filter[NORTHEAST];

        //divide with the sum of all filters
        pixels_sum /= final_filter;
        // get 8 righter bits because we need <=255 demical value
        final_buffer[cursor] = (char)pixels_sum & 0xFF;
        

        // southeast corner pixel
        // pixels_sum of the current rank 
        cursor = block_img_height*block_img_width-1;
        pixels_sum = initial_buffer[cursor] * filter[CENTER];
        pixels_sum += initial_buffer[cursor-1] * filter[EAST];
        pixels_sum += initial_buffer[cursor-block_img_width] * filter[SOUTH];
        pixels_sum += initial_buffer[cursor-block_img_width-1] * filter[SOUTHEAST];

        // pixels that have received from other ranks
        pixels_sum += south_row[block_img_width-1] * filter[NORTH];
        pixels_sum += south_row[block_img_width-2] * filter[NORTHEAST];
        pixels_sum += east_column[block_img_height-1] * filter[WEST];
        pixels_sum += east_column[block_img_height-2] * filter[SOUTHWEST];
        pixels_sum += southeast_corner * filter[NORTHWEST];

        //divide with the sum of all filters
        pixels_sum /= final_filter;
        // get 8 righter bits because we need <=255 demical value
        final_buffer[cursor] = (char)pixels_sum & 0xFF;

        // wait for all sends to process
        // wait all 

	
        MPI_Waitall(8, send_request, MPI_STATUSES_IGNORE);

        // swap initial_buffer with final_buffer for next round
        unsigned char * temp_buffer = final_buffer;
        final_buffer = initial_buffer;
        initial_buffer = temp_buffer;

        // checking for convolution every 20 reps 
        if (((i+1) % 20) == 0)
        {

            unsigned char rank_contrast, pixel_contrast = 0;
            // pixel_contrast == 0 means that hasn't changed after filtering it
            for (int i = 0; i < block_img_width * block_img_height; i++)
            {
                if (initial_buffer[i] != final_buffer[i]) 
                {
                    pixel_contrast = 1;
                    break;
                }
            }
            // sending to root and takes the max value of the total ranks
            // rank_contrast is the receive buffer

		
            MPI_Reduce(&pixel_contrast, &rank_contrast, 1, MPI_CHAR, MPI_MAX, 0, cart);
            MPI_Bcast(&rank_contrast, 1, MPI_CHAR, 0, cart);
            // if the max value is 0, all threads/ranks sent 0 so the image hasn't changed after filtering it
            if (rank_contrast == 0)
            {
                break;
            }
        }
    }

    free(north_row);
    free(west_column);
    free(east_column);
    free(south_row);

    return initial_buffer;
}



// same as above, but for rgb , rgb needs 3 bytes for every pixel
unsigned char * convolution_function_rgb(int block_img_width, int block_img_height, unsigned char *filter, int array_ranks[], int reps, MPI_Comm cart,unsigned char *initial_buffer, unsigned char *final_buffer) 
{   
    // Allocating space for neighbouring pixels
    unsigned char * north_row = (unsigned char*)malloc(block_img_width);
    unsigned char * south_row = (unsigned char*)malloc(block_img_width);

    // either we multiply height by 3 or width for rgb
    unsigned char * west_column = (unsigned char*)malloc(block_img_height * 3);
    unsigned char * east_column = (unsigned char*)malloc(block_img_height * 3);

    // northwest_corner -> north west corner, se -> southeast corner etc...
    unsigned char northwest_corner[3], northeast_corner[3], southwest_corner[3], southeast_corner[3];

    //  9 = 4 sides + 4 corners + center
    MPI_Request send_request[9], receive_request[9];

    // sum of all filters
    unsigned int final_filter = 0;
    for (int i = 0; i < 9; i++)
    {
        final_filter += filter[i];
    }

    // For every rep
    for (int i = 0; i < reps; i++) 
    {
        // send pixels to other ranks, 4 corners , 4 sides       
        // sending one element AND MPI_CHAR we get a corner & row. Sending one element AND type_column we get a column(side)

        // intial_buffer[0] = northwest , north , west
        MPI_Isend(&initial_buffer[0], 3, MPI_CHAR, array_ranks[NORTHWEST], NORTHWEST, cart, &send_request[NORTHWEST]);
        MPI_Isend(&initial_buffer[0], block_img_width, MPI_CHAR, array_ranks[NORTH], NORTH, cart, &send_request[NORTH]);
        MPI_Isend(&initial_buffer[0], 1, type_column_rgb, array_ranks[WEST], WEST, cart, &send_request[WEST]);

        // initial_buffer[block_img_width-3]= northeast , east
        MPI_Isend(&initial_buffer[block_img_width-3], 3, MPI_CHAR, array_ranks[NORTHEAST], NORTHEAST, cart, &send_request[NORTHEAST]);     
        MPI_Isend(&initial_buffer[block_img_width-3], 1, type_column_rgb, array_ranks[EAST], EAST, cart, &send_request[EAST]);

        // initial_buffer[(block_img_height-1)*block_img_width] = southwest , south
        MPI_Isend(&initial_buffer[(block_img_height-1)*block_img_width], 3, MPI_CHAR, array_ranks[SOUTHWEST], SOUTHWEST, cart, &send_request[SOUTHWEST]);
        MPI_Isend(&initial_buffer[(block_img_height-1)*block_img_width], block_img_width, MPI_CHAR, array_ranks[SOUTH], SOUTH, cart, &send_request[SOUTH]);

        // initial_buffer[block_img_height*block_img_width-3] = southeast
        MPI_Isend(&initial_buffer[block_img_height*block_img_width-3], 3, MPI_CHAR, array_ranks[SOUTHEAST], SOUTHEAST, cart, &send_request[SOUTHEAST]);

        // center is calculated independently
        send_request[CENTER] = MPI_REQUEST_NULL;

        // get pixels from other ranks, 4 corner, 4 sides
        // northwest , north , west
        MPI_Irecv(&northwest_corner, 3, MPI_CHAR, array_ranks[NORTHWEST], SOUTHEAST, cart, &receive_request[NORTHWEST]);
        MPI_Irecv(north_row, block_img_width, MPI_CHAR, array_ranks[NORTH], SOUTH, cart, &receive_request[NORTH]);
        MPI_Irecv(west_column, block_img_height * 3, MPI_CHAR, array_ranks[WEST], EAST, cart, &receive_request[WEST]);

        // northeast , east
        MPI_Irecv(&northeast_corner, 3, MPI_CHAR, array_ranks[NORTHEAST], SOUTHWEST, cart, &receive_request[NORTHEAST]);     
        MPI_Irecv(east_column, block_img_height * 3, MPI_CHAR, array_ranks[EAST], WEST, cart, &receive_request[EAST]);

        // southwest , south
        MPI_Irecv(&southwest_corner, 3, MPI_CHAR, array_ranks[SOUTHWEST], NORTHEAST, cart, &receive_request[SOUTHWEST]);
        MPI_Irecv(south_row, block_img_width, MPI_CHAR, array_ranks[SOUTH], NORTH, cart, &receive_request[SOUTH]);

        // southeast
        MPI_Irecv(&southeast_corner, 3, MPI_CHAR, array_ranks[SOUTHEAST], NORTHWEST, cart, &receive_request[SOUTHEAST]);

        // center is calculated independently
        receive_request[CENTER] = MPI_REQUEST_NULL;

        // calculate all inside pixels using OPENMP
        #ifdef _OPENMP
        #pragma omp parallel for num_threads(THREADNUM)
        #endif

        for (int i = 1; i < block_img_height-1; i++)
        {
            for (int l = 3; l < block_img_width-3; l ++) 
            {

                // find offset of current pixel in array
                // for example width= 25 , height=50 =1250pixels , -> last pixel 49*25 + 25 = 1250
                unsigned int offset_current_pixel = i*block_img_width + l;

                // for inside pixels, just sum all 9 pixels(the grid 3x3) multiplied by their coefficients

                unsigned int pixels_sum = northern_row(offset_current_pixel,3,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += center_row(offset_current_pixel,3,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += southern_row(offset_current_pixel,3,block_img_width,block_img_height,filter,initial_buffer);

                // divide with the sum of all filters
                pixels_sum /= final_filter;

                // get 8 righter bits
                final_buffer[offset_current_pixel] = (char)pixels_sum & 0xFF;
            }
        }
            
        // wait until we receive all needed pixels for pixels on the edges
        MPI_Waitall(8, receive_request, MPI_STATUSES_IGNORE);

        // calculating pixels('grid-3x3') on northern side
        for (int i = 3; i < block_img_width-3; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 2nd and 3rd row of the current 'grid'

            unsigned int pixels_sum = center_row(i,3,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += southern_row(i,3,block_img_width,block_img_height,filter,initial_buffer);

            // calculating pixels received from the above rank ( 1st row of the 'grid')
            // for example : these pixels are the southest pixels of the above rank that got connected using send/receive requests
            pixels_sum += north_row[i-3] * filter[SOUTHEAST];
            pixels_sum += north_row[i] * filter[SOUTH];
            pixels_sum += north_row[i+3] * filter[SOUTHWEST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[i] = (char)pixels_sum & 0xFF;
        }

        // calculating pixels('grid-3x3') on southest side
        unsigned int last_line = (block_img_height-1)*block_img_width;
        for (int i = 3; i < block_img_width-3; i++) 
        {
            // Calculating pixels at the current block(from the current rank) 
            // for example : calculating the 1st and 2nd row of the current 'grid'

            unsigned int pixels_sum = northern_row(last_line+i,3,block_img_width,block_img_height,filter,initial_buffer);
            pixels_sum += center_row(last_line+i,3,block_img_width,block_img_height,filter,initial_buffer);

            // calculating pixels received from the below rank ( 3nd row of the 'grid')
            // for example : these pixels are the northern pixels of the below rank that got connected using send/receive requests
            pixels_sum += south_row[i-3] * filter[NORTHEAST];
            pixels_sum += south_row[i] * filter[NORTH];
            pixels_sum += south_row[i+3] * filter[NORTHWEST];

            // divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits
            final_buffer[last_line+i] = (char)pixels_sum & 0xFF;
        }

        unsigned int cursor = 0;
        unsigned int pixels_sum = 0;

        // process corners and left and right edges for each rgb_current_color
        for (unsigned char rgb_current_color = 0; rgb_current_color < 3; rgb_current_color++) 
        {
            // calculating pixels('grid-3x3') on western side
            for (unsigned int i = 1; i < block_img_height-1; i++)
            {
                // Calculating pixels at the current block(from the current rank) 
                // for example : calculating the 2nd and 3rd column of the current 'grid'
                cursor = i*block_img_width + rgb_current_color;
                pixels_sum = center_column(cursor,3,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += eastern_column(cursor,3,block_img_width,block_img_height,filter,initial_buffer);

                // calculating pixels received from the left rank ( 1st column of the 'grid')
                // for example : these pixels are the eastern pixels of the left rank that got connected using send/receive requests
                pixels_sum += west_column[(i-1)*3 + rgb_current_color] * filter[SOUTHEAST];
                pixels_sum += west_column[i*3 + rgb_current_color] * filter[EAST];
                pixels_sum += west_column[(i+1)*3 + rgb_current_color] * filter[NORTHEAST];

                // divide with the sum of all filters
                pixels_sum /= final_filter;
                // get 8 righter bits
                final_buffer[cursor] = (char)pixels_sum & 0xFF;
            }

            // calculating pixels('grid-3x3') on eastern side
            for (unsigned int i = 1; i < block_img_height-1; i++) 
            {
                // Calculating pixels at the current block(from the current rank) 
                // for example : calculating the 1st and 2nd column of the current 'grid'

                cursor = (i+1)*block_img_width - 3 + rgb_current_color;
                pixels_sum = center_column(cursor,3,block_img_width,block_img_height,filter,initial_buffer);
                pixels_sum += western_column(cursor,3,block_img_width,block_img_height,filter,initial_buffer);;

                // calculating pixels received from the right rank ( 3rd column of the 'grid')
                // for example : these pixels are the eastern pixels of the right rank that got connected using send/receive requests
                pixels_sum += east_column[(i-1)*3 + rgb_current_color] * filter[SOUTHWEST];
                pixels_sum += east_column[i*3 + rgb_current_color] * filter[WEST];
                pixels_sum += east_column[(i+1)*3 + rgb_current_color] * filter[NORTHWEST];

                // divide with the sum of all filters
                pixels_sum /= final_filter;

                // get 8 righter bits
                final_buffer[cursor] = (char)pixels_sum & 0xFF;
            }

            // calculating corner pixels
            // northwest corner pixel
            // pixels_sum of the current rank 
            pixels_sum = initial_buffer[rgb_current_color] * filter[CENTER];
            pixels_sum += initial_buffer[rgb_current_color+3] * filter[WEST];
            pixels_sum += initial_buffer[rgb_current_color+block_img_width] * filter[NORTH];
            pixels_sum += initial_buffer[rgb_current_color+block_img_width+3] * filter[NORTHWEST];

            // pixels that have received from other ranks
            pixels_sum += north_row[rgb_current_color] * filter[SOUTH];
            pixels_sum += north_row[rgb_current_color+3] * filter[SOUTHWEST];
            pixels_sum += west_column[rgb_current_color] * filter[EAST];
            pixels_sum += west_column[rgb_current_color+3] * filter[NORTHEAST];
            pixels_sum += northwest_corner[rgb_current_color] * filter[SOUTHEAST];

            //divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits because we need <=255 demical pixels_sumu 
            final_buffer[rgb_current_color] = (char)pixels_sum & 0xFF;


            // northeast corner pixel
            // pixels_sum of the current rank 
            cursor = block_img_width+rgb_current_color-3;
            pixels_sum = initial_buffer[cursor] * filter[CENTER];
            pixels_sum += initial_buffer[cursor-3] * filter[EAST];
            pixels_sum += initial_buffer[cursor+block_img_width] * filter[NORTH];
            pixels_sum += initial_buffer[cursor+block_img_width-3] * filter[NORTHEAST];

            // pixels that have received from other ranks
            pixels_sum += north_row[cursor] * filter[SOUTH];
            pixels_sum += north_row[cursor-3] * filter[SOUTHEAST];
            pixels_sum += east_column[rgb_current_color] * filter[WEST];
            pixels_sum += east_column[rgb_current_color+3] * filter[NORTHWEST];
            pixels_sum += northeast_corner[rgb_current_color] * filter[SOUTHWEST];

            //divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits because we need <=255 demical pixels_sum
            final_buffer[cursor] = (char)pixels_sum & 0xFF;


            // southwest corner pixel
            // pixels_sum of the current rank 
            cursor = (block_img_height-1)*block_img_width+rgb_current_color;
            pixels_sum = initial_buffer[cursor] * filter[CENTER];
            pixels_sum += initial_buffer[cursor+3] * filter[WEST];
            pixels_sum += initial_buffer[cursor-block_img_width] * filter[SOUTH];
            pixels_sum += initial_buffer[cursor-block_img_width+3] * filter[SOUTHWEST];

            // pixels that have received from other ranks
            pixels_sum += south_row[rgb_current_color] * filter[NORTH];
            pixels_sum += south_row[rgb_current_color+3] * filter[NORTHWEST];
            pixels_sum += west_column[block_img_height+rgb_current_color-3] * filter[EAST];
            pixels_sum += west_column[block_img_height+rgb_current_color-6] * filter[SOUTHEAST];
            pixels_sum += southwest_corner[rgb_current_color] * filter[NORTHEAST];

            //divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits because we need <=255 demical pixels_sum
            final_buffer[cursor] = (char)pixels_sum & 0xFF;

            // southeast corner pixel
            // pixels_sum of the current rank 
            cursor = block_img_height*block_img_width+rgb_current_color-3;
            pixels_sum = initial_buffer[cursor] * filter[CENTER];
            pixels_sum += initial_buffer[cursor-3] * filter[EAST];
            pixels_sum += initial_buffer[cursor-block_img_width] * filter[SOUTH];
            pixels_sum += initial_buffer[cursor-block_img_width-3] * filter[SOUTHEAST];

            // pixels that have received from other ranks
            pixels_sum += south_row[block_img_width+rgb_current_color-3] * filter[NORTH];
            pixels_sum += south_row[block_img_width+rgb_current_color-6] * filter[NORTHEAST];
            pixels_sum += east_column[block_img_height+rgb_current_color-3] * filter[WEST];
            pixels_sum += east_column[block_img_height+rgb_current_color-6] * filter[SOUTHWEST];
            pixels_sum += southeast_corner[rgb_current_color] * filter[NORTHWEST];

            //divide with the sum of all filters
            pixels_sum /= final_filter;

            // get 8 righter bits because we need <=255 demical pixels_sum
            final_buffer[cursor] = (char)pixels_sum & 0xFF;
        }

        // wait for all sends to process
        // wait all
        MPI_Waitall(8, send_request, MPI_STATUSES_IGNORE);

        // swap initial_buffer with final_buffer for next round
        unsigned char * temp_buffer = final_buffer;
        final_buffer = initial_buffer;
        initial_buffer = temp_buffer;

        // checking for convolution every 20 reps 
        if (((i+1) % 20) == 0)
        {

            unsigned char rank_contrast, pixel_contrast = 0;
            // pixel_contrast == 0 means that hasn't changed after filtering it
            for (int i = 0; i < block_img_width * block_img_height; i++)
            {
                if (initial_buffer[i] != final_buffer[i]) 
                {
                    pixel_contrast = 1;
                    break;
                }
            }
            // sending to root and takes the max value of the total ranks
            // rank_contrast is the receive buffer
            MPI_Reduce(&pixel_contrast, &rank_contrast, 1, MPI_CHAR, MPI_MAX, 0, cart);
            MPI_Bcast(&rank_contrast, 1, MPI_CHAR, 0, cart);
            // if the max value is 0, all threads/ranks sent 0 so the image hasn't changed after filtering it
            if (rank_contrast == 0)
            {
                break;
            }
        }

    }

    free(north_row);
    free(west_column);
    free(east_column);
    free(south_row);

    return initial_buffer;
}
