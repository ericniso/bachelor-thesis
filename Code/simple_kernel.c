__global__
void my_kernel()
{
    // codice GPU...
}

// CPU code
int main()
{
    int n_blocks = 8; // Numero di thread block
    int n_threads_per_block = 32; // Numero di thread per blocco

    // Esecuzione kernel
    my_kernel<<<n_blocks, n_threads_per_block>>>();
}
