__global__
void
my_kernel()
{
    // Computing current thread global id
    int id = threadIdx.x + blockIdx.x * blockDim.x;
}
