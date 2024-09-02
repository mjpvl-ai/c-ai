# Define the compiler
CC = gcc

# Define the flags for compilation
CFLAGS = -Wall -Wextra -O2

# Define the name of the output executable
TARGET = example

# Define the source files
SRCS = example.c supervised_learning.c

# Define the object files
OBJS = $(SRCS:.c=.o)

# Define the math library for linking
LIBS = -lm

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

# Rule to build object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up build artifacts
clean:
	rm -f $(TARGET) $(OBJS)
