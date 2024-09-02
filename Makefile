# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -O2

# Directories
OBJ_DIR = obj
BIN_DIR = bin

# Source files
SRCS = $(wildcard *.c)
# Object files corresponding to source files
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)

# Executable
TARGET = $(BIN_DIR)/ai_library_test

# Rules
all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lm

# Compile source files to object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
