# Makefile for hw2 SIFT (MPI+OpenMP)
# Build with mpicxx and enable USE_MPI for rank-aware run path
CXX = mpicxx
CXXFLAGS = -std=c++17 -O3 -fopenmp -march=native -mtune=native -funroll-loops -DUSE_MPI
INCLUDES = -I.

# Source files
SOURCES = hw2.cpp sift.cpp image.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = hw2

# STB image library files
STB_HEADERS = stb_image.h stb_image_write.h

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile
hw2.o: hw2.cpp sift.hpp image.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c hw2.cpp

sift.o: sift.cpp sift.hpp image.hpp $(STB_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c sift.cpp

image.o: image.cpp image.hpp $(STB_HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c image.cpp

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET) result.jpg

.PHONY: all clean
