#include <Kokkos_Core.hpp>
#include <iostream>
#include <cuda_profiler_api.h>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard KokkosScopeGuard;
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  const size_t NP = 460000000;

{
  Kokkos::View<double*, Kokkos::CudaSpace> da ("da", NP);

  //warm up
  for(size_t i=0; i<10; i++)
  {
    Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) 
    {
      double dummy = 123.456;
      da(n) = dummy + 123.456*dummy;
    });
  }

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;
  
  Kokkos::fence();
  auto start_clock=Time::now();
  
  //Kokkos::View<double*, Kokkos::CudaSpace> dummy ("dummy", NP);

  cudaProfilerStart();

  for(size_t i=0; i<10; i++)
  {
    Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) 
    {
      double dummy = 123.456;
      da(n) = dummy + 123.456*dummy;
    });
  }

  Kokkos::fence();

  cudaProfilerStop();

    auto finish_clock = Time::now();
    fsec fs = finish_clock - start_clock;
    std::cout<<"time taken is (msecs):" << fs.count()*1e3 << std::endl;
  
  }
  return 0;
}
