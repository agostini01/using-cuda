NVARCHSUPPORT = -arch=sm_30 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_70,code=compute_70


all : app

app : main.cu
	nvcc $(NVARCHSUPPORT) $^ -o $@ 

.PHONY: clean
clean:
	rm -f app