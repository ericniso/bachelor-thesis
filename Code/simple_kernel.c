__global__
void
my_kernel()
{
    // GPU code...
}

// CPU code
int main()
{

    // Invoke Kernel on GPU
    my_kernel<<<8, 32>>>();

    return 0;
}
