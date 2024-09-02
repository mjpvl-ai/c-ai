# Makefile for compiling ai_program

# Compiler
CC = gcc

# Compiler flags
# CFLAGS = -Wall -Wextra -Werror -O2

# Linker flags
LDFLAGS = -lm

# Source files
SOURCES = analytics.c \
          activation/activation_functions.c \
          Basic_Linear_Algebra_Operations/dot_product.c \
          Basic_Linear_Algebra_Operations/loss_functions.c \
          Basic_Linear_Algebra_Operations/vector_ops.c \
          data_preprocessing/data_preprocessing.c \
          matrix_operations/matrix_determinant.c \
          matrix_operations/matrix_inversion.c \
          matrix_operations/matrix_ops.c \
          matrix_operations/matrix_transpose.c \
          momentum/momentum.c \
          neural_networks/dense_layer.c \
          neural_networks/neural_network_training.c \
          neural_networks/neural_networks.c \
          optimization/adam.c \
          optimization/gradient_descent.c \
          optimization/learning_rate_schedule.c \
          regularization/regularization.c \
          statistics/statistics.c \
          supervised_learning/supervised_learning.c \
          unsupervised_learning/unsupervised_learning.c \
          neural_networks/simple_neural_network.c

# Output binary
TARGET = ai_program

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
