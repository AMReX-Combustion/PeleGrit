#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

__global__ void parallel_for(int n) {
  for(int i=0; i<n; i++)
  {
    double dummy0=1.0;
    double dummy1=dummy0+23232.434*dummy0;
  }
}

int main()
{
  const int N = 460000000;
  int blockSize = 256;
  int numBlocks = (N + blockSize -1) / blockSize;

  //warm up
  for(int j=0; j<100; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(N);
  }

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;

  cudaDeviceSynchronize();
  auto start_clock = Time::now();

  cudaProfilerStart();

  for(int j=0; j<10; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(460000000);
  }
  
  cudaDeviceSynchronize();
  
  cudaProfilerStop();

  auto finish_clock = Time::now();
  fsec fs = finish_clock - start_clock;
  std::cout << "time taken for cuda parallel fro (msecs):" << fs.count()*1e3 << std::endl;

  return 0; 
}
