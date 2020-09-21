#include <iostream>
#include <chrono>
#include <cuda_profiler_api.h>

__global__ void parallel_for(const int n, double* dax, double* dbx,
                             const double dt) {
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid < n) {
        dax[tid] = dax[tid] + dbx[tid]*dt;
    }
}

int main()
{
  const int Nl = 1000;
  const double dt=0.001;
  const int N = 1000000;
  int blockSize = 64;
  int numBlocks = (N + blockSize -1) / blockSize;

  double* dax;
  double* dbx;

  cudaMalloc((void**)&dax, sizeof(double)*N);
  cudaMalloc((void**)&dbx, sizeof(double)*N);

  //warm up
  for(int j=0; j<100; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(N, dax, dbx, dt);
  }

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;

  cudaDeviceSynchronize();
  auto start_clock = Time::now();

  cudaProfilerStart();

  for(int j=0; j<Nl; j++)
  {
    parallel_for<<<numBlocks, blockSize>>>(N, dax, dbx, dt);
  }
  
  cudaDeviceSynchronize();
  
  cudaProfilerStop();

  auto finish_clock = Time::now();
  fsec fs = finish_clock - start_clock;
  std::cout << "time taken for cuda parallel for (msecs):" << fs.count()*1e3/Nl << std::endl;

  cudaFree(dax);
  cudaFree(dbx);

  return 0; 
}
