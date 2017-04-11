################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/DataLayer.cpp \
../src/FullConnLayer.cpp \
../src/Layer.cpp \
../src/ZeroNet.cpp 

OBJS += \
./src/DataLayer.o \
./src/FullConnLayer.o \
./src/Layer.o \
./src/ZeroNet.o 

CPP_DEPS += \
./src/DataLayer.d \
./src/FullConnLayer.d \
./src/Layer.d \
./src/ZeroNet.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/home/zys/local/cuda/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


