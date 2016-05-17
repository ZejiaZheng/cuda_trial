#include <stdio.h>
#include <time.h>
#include "timer.h"
#include "utils.h"

const int N = 1024;  // we are transposing a N by N mat
const int K = 1;

void fill_matrix(float* mat){
    for (int i = 0; i < N*N; i++){
        mat[i] = (float)i;
    }
}

int compare_matrices(float* mat1, float* mat2){
    for (int i = 0; i < N*N; i++){
        if(mat1[i] != mat2[i]) return 0;
    }
    return 1;
}

void transpose_cpu(float* mat, float* mat_new){
    for (int i = 0; i < N; i++){
        for (int j = 0; j< N; j++){
            mat_new[j*N + i] = mat[i*N + j];
        }
    }
}

__global__ void transpose_serial(float* mat, float* mat_new){
    for (int i = 0; i < N; i++){
        for (int j = 0; j< N; j++){
            mat_new[j*N + i] = mat[i*N + j];
        }
    }
}

__global__ void transpose_parallel_row(float* mat, float* mat_new){
    int i = threadIdx.x;

    for (int j = 0; j < N; j++){
        mat_new[j+i*N] = mat[i+j*N];
    }
}

int main(int argc, char const *argv[])
{
    size_t mat_size = N * N * sizeof(float);
    float* mat = (float *) malloc(mat_size);
    float* mat_new = (float *) malloc(mat_size);
    float* mat_gold = (float *) malloc(mat_size);

    fill_matrix(mat);

    clock_t start = clock(), diff;
    transpose_cpu(mat, mat_gold);
    diff = clock() - start;
    printf("CPU Transpose time taken: %f milliseconds \n", diff);


    float *d_in, *d_out;

    cudaMalloc(&d_in, mat_size);
    cudaMalloc(&d_out, mat_size);
    cudaMemcpy(d_in, mat, mat_size, cudaMemcpyHostToDevice);

    GpuTimer timer;
    
    // transpose serial
    timer.Start();
    transpose_serial<<<1,1>>>(d_in, d_out);
    timer.Stop();

    cudaMemcpy(d_out, mat_new, mat_size, cudaMemcpyDeviceToHost);

    printf("transpose_serial: %g ms.\nVerifying transpose...%s\n", timer.Elapsed(), compare_matrices(mat_new, mat_gold) ? "Failed" : "Success");

    // transpose parallel row
    timer.Start();
    transpose_parallel_row<<<1,N>>>(d_in, d_out);
    timer.Stop();

    cudaMemcpy(d_out, mat_new, mat_size, cudaMemcpyDeviceToHost);

    printf("transpose_parallel_row: %g ms.\nVerifying transpose...%s\n", timer.Elapsed(), compare_matrices(mat_new, mat_gold) ? "Failed" : "Success");


    // show orignal mat, for debug purposes
    /*printf("original_mat: \n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%3.1f ", mat[i*N + j]);
        }
        printf("\n");
    }

    printf("transposed_mat: \n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            printf("%3.1f ", mat_gold[i*N + j]);
        }
        printf("\n");
    }*/
    cudaFree(d_in);
    cudaFree(d_out);
    free(mat);
    free(mat_gold);

    return 0;
}