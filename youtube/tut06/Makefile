# NVARCHSUPPORT = -arch=sm_30 \
# 	-gencode=arch=compute_50,code=sm_50 \
# 	-gencode=arch=compute_52,code=sm_52 \
# 	-gencode=arch=compute_60,code=sm_60 \
# 	-gencode=arch=compute_61,code=sm_61 \
# 	-gencode=arch=compute_70,code=sm_70 \
# 	-gencode=arch=compute_70,code=compute_70

NVARCHSUPPORT = -arch=sm_30

NVCFLAGS = -Xcompiler -Ofast
OBJS=FindClosestCPU.o FindClosestGPU.o main.o

all : app

app : $(OBJS)
	nvcc $(NVARCHSUPPORT) $^ -o $@ 

FindClosestGPU.o : FindClosestGPU.cu FindClosestGPU.h config.h
	nvcc -o $@ $(NVARCHSUPPORT) $(NVCFLAGS) -c $<

FindClosestCPU.o : FindClosestCPU.cc FindClosestCPU.h
	nvcc -o $@ $(NVCFLAGS) -c $<

main.o : main.cc config.h
	nvcc -o $@ $(NVCFLAGS) -c $<

.PHONY: clean
clean:
	rm -f app *.o *.prof