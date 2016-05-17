#include <stdio.h>
#include "timer.h"
#include "utils.h"

const int N = 8;  // we are transposing a N by N mat
const int K = 1;

void fill_matrix(float* mat){
    for (int i = 0; i < N*N; i++){
        mat[i] = (float)i;
    }
}

int main(int argc, char const *argv[])
{
    size_t mat_size = N * N * sizeof(float);
    float* mat = (float *) malloc(mat_size);

    fill_matrix(mat);

    // show orignal mat, for debug purposes
    printf("original_mat: \n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%3f", mat[i*N + j]);
        }
    }

    return 0;
}