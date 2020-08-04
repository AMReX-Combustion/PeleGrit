# AMReX
DIM = 2
COMP = gnu
PRECISION = DOUBLE

BL_NO_FORT = TRUE

# Profiling
PROFILE = FALSE
TINY_PROFILE = TRUE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = FALSE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PeleC
USE_REACT = FALSE
USE_EB = FALSE
Eos_dir := Fuego
Reactions_dir := Fuego
Chemistry_Model := decane_3sp
Transport_dir := Simple

# PeleC-MP
USE_PARTICLES = TRUE
SPRAY_FUEL_NUM = 1
USE_GRIT = TRUE

# GNU Make
Bpack := ./Make.package
Blocs := .
include $(PELEC_HOME)/ExecCpp/Make.PeleC

#link to grit
#LIBRARIES += -L$(GRIT_LIB) -lgritpeleinterface
LIBRARIES += -L$(GRIT_LIB) -lgritpelesprayinterface
LIBRARIES += -L$(CUDA_LIB) -lcudart
LIBRARIES += -L$(KOKKOS_LIB) -lkokkos
LIBRARIES += -ldl
