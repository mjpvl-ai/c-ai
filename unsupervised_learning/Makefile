# Makefile

# Compiler
CC = gcc

# Compiler Flags
CFLAGS = -Wall -Wextra -std=c11

# Libraries
LIBS = -lm

# Target executable
TARGET = example

# Source files
SRCS = example.c unsupervised_learning.c

# Object files (derived from source files)
OBJS = $(SRCS:.c=.o)

# Default rule
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)

# Rule to build the object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up the generated files
clean:
	rm -f $(OBJS) $(TARGET)
