SRCDIR= ./src
OBJDIR=./obj
TEST_SRCDIR=./test
TEST_OBJDIR=./obj/test
SUFFIX=

EXE=Zero
TEST_EXE =TestZero
CPP = g++ -std=c++11
NVCC=nvcc
CPPFLAGS= -Wall -g
NVCCFLAGS= 

SRCS= $(shell find  $(SRCDIR) -name "*.cpp" )
CUDASRCS= $(shell find $(SRCDIR) -name "*.cu")
OBJS= $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o$(SUFFIX)) $(CUDASRCS:$(SRCDIR)/%.cu=$(OBJDIR)/%.co$(SUFFIX))

TEST_SRCS= $(shell find $(TEST_SRCDIR) -name "*.cpp")
TEST_CUDASRCS=$(shell find $(TEST_SRCDIR) -name "*.cu")
TEST_OBJS= $(TEST_SRCS:$(TEST_SRCDIR)/%.cpp=$(TEST_OBJDIR)/%.o$(SUFFIX)) $(TEST_CUDASRCS:$(TEST_SRCDIR)/%.cu=$(TEST_OBJDIR)/%.co$(SUFFIX))


INCLUDE = -I ./include  -I /home/zys/local/cuda/include -I /usr/local/boost -I ./test -I ./include/tools
LIBS=-L /usr/lib/x86_64-linux-gnu \
	 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_gpu -lopencv_ts -lopencv_video -lopencv_objdetect -lopencv_ml -lpthread \
	 -lcudart -lcurand -L /home/zys/local/cuda/lib64  -lcudnn -L /home/zys/local/cuda/lib64 -lcublas -L /home/zys/local/cuda/lib64 \
	 -lglog \
	 -lprotobuf \
	 -lboost_system -lboost_thread -L /usr/lib/x86_64-linux-gnu \
	 -fopenmp

all:zero test

zero:$(OBJDIR)/Zero.o $(OBJS)
	@echo 'creating binary "$(EXE)"'
	@$(CPP) $(CPPFLAGS) -o $(EXE) $^ $(LIBS) 
	@echo 'done...'

test:$(TEST_OBJS)
	@echo "$(TEST_OBJS)"
	@echo 'creating binary "$(TEST_EXE)"'
	@$(CPP) $(CPPFLAGS) -o $(TEST_EXE) $^ $(LIBS) 
	@echo 'done...'

$(OBJDIR)/Zero.o:Zero.cpp
	@echo 'compiling object file "$@" ...'
	@$(CPP) $(CPPFLAGS) -c -o $@  $< $(INCLUDE)

$(OBJDIR)/%.o$(SUFFIX): $(SRCDIR)/%.cpp
	@echo 'compiling object file "$@" ...'
	@$(CPP) $(CPPFLAGS) -c -o $@  $< $(INCLUDE)

$(OBJDIR)/%.co$(SUFFIX): $(SRCDIR)/%.cu
	@echo 'compiling object file "$@" ...'
	@$(NVCC) $(NVCCFLAGS) -c -o $@  $< $(INCLUDE)
	
$(TEST_OBJDIR)/%.o$(SUFFIX):$(TEST_SRCDIR)/%.cpp
	@echo 'compiling object file "$@" ...'
	@$(CPP) $(CPPFLAGS) -c -o $@  $< $(INCLUDE)
	
$(TEST_OBJDIR)/%.co$(SUFFIX): $(TEST_SRCDIR)/%.cu
	@echo 'compiling object file "$@" ...'
	@$(NVCC) $(NVCCFLAGS) -c -o $@  $< $(INCLUDE)

clean:
	@echo 'clear all obj'
	@rm -f $(OBJDIR)/*.o  $(OBJDIR)/tools/*.o $(TEST_OBJDIR)/*.o $(EXE) $(TEST_EXE) $(TEST_OBJDIR)/*.co

