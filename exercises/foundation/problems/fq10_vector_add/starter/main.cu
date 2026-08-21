__global__ void vector_add_kernel(const float* left, const float* right,
                                  float* result, int size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: guard index and write one output element per thread.
  (void)left;
  (void)right;
  (void)result;
  (void)size;
}
