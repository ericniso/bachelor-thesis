__global__
void
my_kernel()
{
    // GPU code...
}

// CPU code
int main()
{
    int n_blocks = 8; // Example blocks number
    int n_threads_per_block = 32; // Example threads per block number

    // Invoke Kernel on GPU
    my_kernel<<<n_blocks, n_threads_per_block>>>();
}
