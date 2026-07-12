MACROS=

CC=gcc
CPP=g++
LANGFLAGS=-std=c++20
COMPFLAGS=-Wextra

ifdef DEBUG_SINGLE_TEST
	MACROS+= -DDEBUG_SINGLE_TEST
	DEBUG=1
endif

ifdef DEBUG
	COMPFLAGS+= -g
	MACROS+= -DDEBUG
else
	COMPFLAGS+= -O3 -fopenmp
endif

ifndef TRUE_REALTIME
	MACROS+= -DUSE_LAST_OPTIMAL_VALUE
endif

SRC=./model_fitting
CCFLAGS=$(LANGFLAGS) $(COMPFLAGS) $(MACROS)
BUILD_DIR=$(SRC)/build/$(CC)
THIRD_PARTY_OBJ=

NVCC=nvcc
MSVCFLAGS=--std=c++20 -Xcompiler /utf-8 -Xcompiler /std:c++20
GCCFLAGS=-Xcompiler -std=c++17 -Xcompiler -fopenmp 
UNAME=$(shell uname)
ifeq ($(UNAME), Linux)
	HOSTFLAGS=$(GCCFLAGS)
	CLEANOBJ=build-clean
	DSOLIB=-ldl
else
	HOSTFLAGS=$(MSVCFLAGS)
	CLEANOBJ=win-clean
	EXEEXT=.exe
endif
NVLANGFLAGS=-DCUDA_ENABLED
NVCOMPFLAGS=-O3 -arch=sm_86 $(HOSTFLAGS)
NVCCFLAGS=$(MACROS) $(NVLANGFLAGS) $(NVCOMPFLAGS)
NVBUILD_DIR=$(SRC)/build/$(NVCC)

SOURCES=$(wildcard $(SRC)/*.cu $(SRC)/*.cpp)
HEADER=$(wildcard $(SRC)/*.h)
BIN_DIR=$(SRC)/bin

CU_OBJS=$(patsubst $(SRC)%, $(BUILD_DIR)%, $(filter %.o,$(SOURCES:.cu=.o)))

NVCU_OBJS=$(patsubst $(SRC)%, $(NVBUILD_DIR)%, $(filter %.o,$(SOURCES:.cu=.o)))

.PHONY : $(CU_OBJS) global_optimizer

$(CU_OBJS) : $(BUILD_DIR)/%.o : $(SRC)/%.cu $(HEADER)
	$(CPP) -c -x c++ $(CCFLAGS) $< -o $@

$(NVCU_OBJS) : $(NVBUILD_DIR)/%.o : $(SRC)/%.cu $(HEADER)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

global_optimizer : $(CU_OBJS)
	$(CPP) $(CCFLAGS) $(CU_OBJS) $(CPP_OBJS) $(THIRD_PARTY_OBJ) -o $(BIN_DIR)/global_optimizer$(EXEEXT) $(DSOLIB)

global_optimizer_cuda: $(NVCU_OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCU_OBJS) -o $(BIN_DIR)/global_optimizer_cuda$(EXEEXT)