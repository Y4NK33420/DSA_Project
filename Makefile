# Compiler
CC = g++

# Compiler flags
CFLAGS = -std=c++11 -Wall

# Include directories
INCLUDES = -Iinclude

# Library directories
LIBS = -Llib -lmingw32 -lSDL2main -lSDL2 -lSDL2_ttf

# Source files
SRCS = src/main.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = build/main

# Default rule
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(OBJS) $(LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean