#include "learning_rate_schedule.h"
#include <math.h>
#include <stdlib.h>

LearningRateSchedule* create_learning_rate_schedule(double initial_learning_rate, double decay_rate, int decay_steps) {
    LearningRateSchedule *schedule = (LearningRateSchedule *)malloc(sizeof(LearningRateSchedule));
    schedule->initial_learning_rate = initial_learning_rate;
    schedule->decay_rate = decay_rate;
    schedule->decay_steps = decay_steps;
    schedule->current_step = 0;
    return schedule;
}

double get_current_learning_rate(LearningRateSchedule *schedule) {
    return schedule->initial_learning_rate * pow(schedule->decay_rate, (double)(schedule->current_step / schedule->decay_steps));
}

void update_learning_rate(LearningRateSchedule *schedule) {
    schedule->current_step++;
}

void free_learning_rate_schedule(LearningRateSchedule *schedule) {
    free(schedule);
}
