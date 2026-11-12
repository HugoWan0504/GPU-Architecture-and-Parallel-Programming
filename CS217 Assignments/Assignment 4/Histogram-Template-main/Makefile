NVCC        = nvcc
ifeq (,$(shell which nvprof 2>/dev/null))
NVCC_FLAGS  = -O3 -lineinfo
else
NVCC_FLAGS  = -O3 -lineinfo --std=c++03
endif
# Real GPU server (modern nvcc): build for the installed GPU's architecture, so one
# Makefile works on every machine (A30 on bender, GTX 950 on the CI runner, ...).
# The GPGPU-Sim container (CUDA 4.0 nvcc) builds for sm_20, the simulated GTX 480: CUDA 4.0's
# default (sm_10) has no atomicAdd.
# Override with e.g. make ARCH=-arch=sm_86 (needed to build on a machine without a GPU)
ifneq (,$(findstring release 4.,$(shell nvcc --version 2>/dev/null)))
ARCH        ?= -arch=sm_20
else
ARCH        ?= -arch=native
endif
NVCC_FLAGS  += $(ARCH)
LD_FLAGS    = -lcudart
EXE	        = histogram

default: basic privatized

main-basic.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

main-privatized.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) -DPRIVATIZED

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

basic: main-basic.o support.o
	$(NVCC) main-basic.o support.o -o $(EXE)-basic $(LD_FLAGS)

privatized: main-privatized.o support.o
	$(NVCC) main-privatized.o support.o -o $(EXE)-privatized $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)-basic $(EXE)-privatized
