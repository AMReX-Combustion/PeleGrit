#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

__global__ void parallel_for(const int n, double* da) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    double dummy = 123.456;
    if (tid < n) {
        da[tid] = dummy + 123.456*dummy;
    }
}

int main()
{
  const int N = 460000000;
  int blockSize = 256;
  int numBlocks = (N + blockSize -1) / blockSize;

  double* da;

  cudaMalloc((void**)&da, sizeof(double)*N);

  //warm up
  for(int j=0; j<10; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(N, da);
  }

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;

  cudaDeviceSynchronize();
  auto start_clock = Time::now();

  cudaProfilerStart();

  for(int j=0; j<10; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(N, da);
  }
  
  cudaDeviceSynchronize();
  
  cudaProfilerStop();

  auto finish_clock = Time::now();
  fsec fs = finish_clock - start_clock;
  std::cout << "time taken for cuda parallel fro (msecs):" << fs.count()*1e3 << std::endl;

  cudaFree(da);

  return 0; 
}
