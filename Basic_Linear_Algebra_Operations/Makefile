# Makefile

CC = gcc
CFLAGS = -Wall -lm

# List of object files
OBJ = dot_product.o loss_functions.o vector_ops.o example.o

# Target executable
TARGET = example

# Rule to build the target executable
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(CFLAGS)

# Rule to build each object file
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

# Clean up object files and the executable
.PHONY: clean
clean:
	rm -f $(OBJ) $(TARGET)
