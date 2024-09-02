# Define the compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11

# Define the target executable and object files
TARGET = example
OBJS = example.o adam.o gradient_descent.o learning_rate_schedule.o

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) -lm

# Rule to build the object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to clean up generated files
clean:
	rm -f $(TARGET) $(OBJS)
