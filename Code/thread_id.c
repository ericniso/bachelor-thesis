__global__
void
my_kernel(int* array)
{
    // Computing current thread global id
    int id = threadIdx.x + (blockIdx.x * blockDim.x) * (gridIdx.x * gridDim.x);

    // Work with array values
    array[id] = 0;
}
