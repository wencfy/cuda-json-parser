CXX = g++
CXXFLAGS = -std=c++20 -Wall -Iinclude
LDFLAGS =

BUILD_DIR = build

SOURCES := $(shell find src -name '*.cpp')
OBJECTS = $(SOURCES:%.cpp=$(BUILD_DIR)/%.o)
DEPS = $(OBJECTS:.o=.d)
TARGET = $(BUILD_DIR)/my_program

all: $(TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD_DIR)

-include $(DEPS)
