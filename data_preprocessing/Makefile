# Makefile

CC = gcc
CFLAGS = -Wall -lm

# Source files
SRC = example.c data_preprocessing.c ../statistics/statistics.c

# Object files
OBJ = $(SRC:.c=.o)

# Target executable
TARGET = example

# Rule to build the target executable
$(TARGET): $(OBJ)
	$(CC) -o $(TARGET) $(OBJ) $(CFLAGS)

# Rule to build each object file
%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)

# Clean up object files and the executable
.PHONY: clean
clean:
	rm -f $(OBJ) $(TARGET)
