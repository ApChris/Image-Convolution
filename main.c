#include <stdio.h>
#include <math.h>
#include "convolution.h"

int main(int argc, char **argv)
{
	//MPI variables
	int rank,size;
	MPI_Comm cart;
	
	// times variables
	double start_time, finish_time, received_time, received_time_max;

	// coefficients 1-> corners , 2-> sides ,4 -> center
	unsigned char filters[9] = {1,2,1,2,4,2,1,2,1};
	
	int img_side;
	int block_img_width;
	int block_img_height;

	// Inform user about parameters 
    if (argc != 6) 
    {
    	printf("After %s add the following information",argv[0]);
        printf("\nImage name\tWidth\tHeight\tRGB -> 1 , BW -> 0\tRepetitions\n");
        return 0;
    }

    // Read parameters
    char *img_name = argv[1];
    unsigned int img_width = atoi(argv[2]);
    unsigned int img_height = atoi(argv[3]);
    unsigned int img_color = argv[4][0] == '1';
    
    unsigned int repetitions = atoi(argv[5]);

    // MPI initialise
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0)
    {
    	printf("image name:%s\nimage width:%d\nimage height:%d\nimage color:%d\nrepetitions:%d\n",img_name,img_width,img_height,img_color,repetitions);

    }

    // If the image is RGB , RGB-> 1 pixel = 3bytes , BW -> 1 pixel = 1byte
    if (img_color == 1)
    {
    	img_width *= 3; 
    }

    // size = number of threads
	
	img_side = (int)sqrt(size);

	// Block width & height
	block_img_width = img_width / img_side;
	block_img_height = img_height / img_side;

	// arrays periodic(true)
    int periodic[2] = {1, 1};
	// array img_side X img_side
	int img_sides[2] = {img_side, img_side};

	


  	// Create a communication
    MPI_Cart_create(MPI_COMM_WORLD, 2, img_sides, periodic, 1, &cart);

    if (cart == MPI_COMM_NULL) 
    {
        printf("Can't create cart communicator\n");

        // END
        MPI_Finalize();
        return 1;
    }

    
    // initializing the first position of each block which later becomes the first position of each rank
    int * position = malloc(size * sizeof(int));
    // number of elements to send of each block
    int * elements_to_send = malloc(size * sizeof(int));
    for (int i=0; i<img_side; i++) 
    {
        for (int j=0; j<img_side; j++) 
        {
            position[i*img_side+j] = i*img_width*block_img_height+j*block_img_width;
            elements_to_send [i*img_side+j] = 1;
        }
    }


     
    MPI_Datatype mpi_type_resized;
    MPI_Datatype mpi_type;

    // creating datatype (vector) to send the blocks
    MPI_Type_vector(block_img_height, block_img_width, img_width, MPI_CHAR, &mpi_type);
    MPI_Type_create_resized( mpi_type, 0, sizeof(char), &mpi_type_resized);
    MPI_Type_commit(&mpi_type_resized);

    // load the image 
	unsigned char * buffer = NULL;
    if (rank == 0) 
    {
        buffer = (unsigned char*)malloc(img_width*img_height);
        FILE *f = fopen(img_name, "rb");
        if (f) {
        fread(buffer, 1, img_width*img_height, f);
        fclose(f);
        } 
        else {
            fprintf(stderr,"fopen error\n");
            exit(1);
        }
    }
	
	// processing the image using buffers

	unsigned char * temp_buffer = (unsigned char*)malloc(img_width*img_height/size);
	unsigned char * temp_buffer2 = (unsigned char*)malloc(img_width*img_height/size);

    // scattering the buffer to process the image

    MPI_Scatterv(buffer, elements_to_send, position, mpi_type_resized, temp_buffer, (img_width*img_height)/size, MPI_CHAR, 0, cart);


    // Creates a vector (strided) datatype  used in the convolution
    column_type(block_img_width, block_img_height);

    // save the ranks of each process' neighbours
    int neighbouring_ranks[9];
    get_neighbours(rank, cart, neighbouring_ranks);

    // Blocks until all processes in the communicator have reached this routine
    MPI_Barrier(cart);
    start_time = MPI_Wtime();

   unsigned char *resolution;

    if (img_color == 0) // BW 
    {
        resolution = convolution_function(block_img_width, block_img_height, (unsigned char *)filters, neighbouring_ranks, repetitions, cart, temp_buffer, temp_buffer2);
    }
    else 				// RGB
    {
        resolution = convolution_function_rgb(block_img_width, block_img_height, (unsigned char *)filters, neighbouring_ranks, repetitions, cart, temp_buffer, temp_buffer2);
    }

    finish_time = MPI_Wtime();
    received_time = finish_time - start_time;

    // Reduces values on all processes to a single value
    MPI_Reduce(&received_time, &received_time_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(resolution, img_width*img_height/size, MPI_CHAR, buffer, elements_to_send, position, mpi_type_resized, 0, cart);

    MPI_Finalize();

    // write the buffer to file
    if (rank == 0) 
    {
        printf("Received Time: %f\n", received_time_max);
        FILE *edited_file = fopen("Edited.raw", "wb");
        fwrite(buffer, 1, img_width*img_height, edited_file);
        fclose(edited_file);
        free(buffer);
    }
    free(temp_buffer);
    free(temp_buffer2);
    free(position);
    free(elements_to_send);
	return 0;
}
// END OF FILE
