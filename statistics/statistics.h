// statistics.h
#ifndef STATISTICS_H
#define STATISTICS_H

double mean(const double *data, int size);
double median(double *data, int size);
double mode(const double *data, int size);
double variance(const double *data, int size);
double standard_deviation(const double *data, int size);
double covariance(const double *data1, const double *data2, int size);
double correlation_coefficient(const double *data1, const double *data2, int size);

#endif // STATISTICS_H
