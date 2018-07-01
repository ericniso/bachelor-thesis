__global__
void my_kernel(int* array)
{
    // Calcol dell'id globale del thread rispetto al kernel
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Elaborazione dell'elemento dell'array corrispondente al thread id
    array[id] = 0;
}
