#ifndef LEARNING_RATE_SCHEDULE_H
#define LEARNING_RATE_SCHEDULE_H

typedef struct {
    double initial_learning_rate;
    double decay_rate;
    int decay_steps;
    int current_step;
} LearningRateSchedule;

LearningRateSchedule* create_learning_rate_schedule(double initial_learning_rate, double decay_rate, int decay_steps);
double get_current_learning_rate(LearningRateSchedule *schedule);
void update_learning_rate(LearningRateSchedule *schedule);
void free_learning_rate_schedule(LearningRateSchedule *schedule);

#endif // LEARNING_RATE_SCHEDULE_H
