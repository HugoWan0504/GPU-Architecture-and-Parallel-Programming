
NVCC        = nvcc
ARCH       ?= native
NVCC_FLAGS  = -O3 -arch=$(ARCH)
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
EXE	        = mat-add
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
