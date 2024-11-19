CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++20 -Wall -Iinclude
NVCCFLAGS = -std=c++17 -Iinclude
LDFLAGS =

BUILD_DIR = build

CPP_SOURCES := $(shell find src -name '*.cpp')
CU_SOURCES := $(shell find src -name '*.cu')

CPP_OBJECTS = $(CPP_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
CU_OBJECTS = $(CU_SOURCES:%.cu=$(BUILD_DIR)/%.o)

OBJECTS = $(CPP_OBJECTS) $(CU_OBJECTS)

TARGET = $(BUILD_DIR)/my_program

all: $(TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(NVCC) $(LDFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d)
