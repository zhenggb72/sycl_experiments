all: segment


segment: segment.cpp 
	mpiicpx -O3 -o segment segment.cpp -lze_loader -lpthread


run: segment
    NEOReadDebugKeys=1 RenderCompressedBuffersEnabled=0 EnablePidFdOrSocketsForIpc=1 mpirun -np 2 ./segment
