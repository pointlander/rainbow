// test.c
#include <stdio.h>
#include <stdlib.h>
extern double __enzyme_autodiff(void*, double*, double*);
double square(double* array) {
    return *array * *array;
}
double dsquare(double* ii, double* shadow) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff((void*) square, ii, shadow);
}
int main() {
    double* ii = (double*)malloc(8);
    double* shadow = (double*)malloc(8);
    for(double i=1; i<5; i++) {
        *ii = i;
        *shadow = 0;
        dsquare(ii, shadow);
        printf("square(%f)=%f, dsquare(%f)=%f\n", i, square(ii), i, *shadow);
    }
}