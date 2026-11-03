
NVCC        = nvcc
ifeq (,$(shell which nvprof 2>/dev/null))
NVCC_FLAGS  = -O3 -lineinfo
else
NVCC_FLAGS  = -O3 -lineinfo --std=c++03
endif
# Real GPU server (modern nvcc): build for the installed GPU's architecture, so one
# Makefile works on every machine (A30 on bender, GTX 950 on the CI runner, ...).
# The GPGPU-Sim container (CUDA 4.0 nvcc) gets no flag.
# Override with e.g. make ARCH=-arch=sm_86 (needed to build on a machine without a GPU)
ifneq (,$(findstring release 4.,$(shell nvcc --version 2>/dev/null)))
ARCH        ?=
else
ARCH        ?= -arch=native
endif
NVCC_FLAGS  += $(ARCH)
ifdef TILE
NVCC_FLAGS += -DTILE_SIZE=$(TILE)
endif
ifdef TILE_X
NVCC_FLAGS += -DTILE_X=$(TILE_X)
endif
ifdef TILE_Y
NVCC_FLAGS += -DTILE_Y=$(TILE_Y)
endif
LD_FLAGS    = -lcudart
EXE	        = sgemm-tiled
OBJ	        = main.o support.o

default: $(EXE)

main.o: main.cu kernel.cu support.h .tile
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

# Records the last block-size settings, so main.o rebuilds when they change
.tile: FORCE
	@echo '$(TILE) $(TILE_X) $(TILE_Y)' | cmp -s - $@ || echo '$(TILE) $(TILE_X) $(TILE_Y)' > $@

FORCE:

clean:
	rm -rf *.o $(EXE) .tile
