#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard KokkosScopeGuard;
  Kokkos::DefaultExecutionSpace::print_configuration(std::cout);

  const size_t NP = 460000000;

  //warm up
  for(size_t i=0; i<100; i++)
  {
    Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) 
    {
      double dummy0 = 1.0;
    });
  }

  typedef std::chrono::high_resolution_clock Time;
  typedef std::chrono::duration<float> fsec;
  
  Kokkos::fence();
  auto start_clock=Time::now();
  
  //Kokkos::View<double*, Kokkos::CudaSpace> dummy ("dummy", NP);

  for(size_t i=0; i<10; i++)
  {
    //Kokkos::View<double*, Kokkos::CudaSpace> dummy ("dummy", NP);
    //Kokkos::View<double*> dummy ("dummy", NP);
    Kokkos::parallel_for(NP, KOKKOS_LAMBDA (const size_t& n) 
    {
      double dummy0 = 1.0;
      double dummy1 = dummy0 + 334.343*dummy0;
    });
  }
    Kokkos::fence();
    auto finish_clock = Time::now();
    fsec fs = finish_clock - start_clock;
    std::cout<<"time taken is (msecs):" << fs.count()*1e3 << std::endl;
  
  return 0;
}
