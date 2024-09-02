# Define the compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11

# Define the target executable and object files
TARGET = example
OBJS = example.o regularization.o ../optimization/adam.o ../optimization/gradient_descent.o ../optimization/learning_rate_schedule.o

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) -lm

# Rule to build object files for example.c
example.o: example.c
	$(CC) $(CFLAGS) -c example.c -o example.o

# Rule to build object files for regularization.c
regularization.o: regularization.c
	$(CC) $(CFLAGS) -c regularization.c -o regularization.o

# Rule to build object files for adam.c
../optimization/adam.o: ../optimization/adam.c
	$(CC) $(CFLAGS) -c ../optimization/adam.c -o ../optimization/adam.o

# Rule to build object files for gradient_descent.c
../optimization/gradient_descent.o: ../optimization/gradient_descent.c
	$(CC) $(CFLAGS) -c ../optimization/gradient_descent.c -o ../optimization/gradient_descent.o

# Rule to build object files for learning_rate_schedule.c
../optimization/learning_rate_schedule.o: ../optimization/learning_rate_schedule.c
	$(CC) $(CFLAGS) -c ../optimization/learning_rate_schedule.c -o ../optimization/learning_rate_schedule.o

# Rule to clean up generated files
clean:
	rm -f $(TARGET) $(OBJS)
