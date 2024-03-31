// test.c
#include <stdio.h>
#include <stdlib.h>
extern double __enzyme_autodiff(void*, double*, double*, size_t);
double square(double* arr, size_t n) {
    for (int i = 0; i < n; i++) {
        arr[i] = arr[i] * arr[i];
    }
    int i, j;
    int swapped = 0;
    for (i = 0; i < n - 1; i++) {
        swapped = 0;
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                double s = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = s;
                swapped = 1;
            }
        }
 
        // If no two elements were swapped by inner loop,
        // then break
        if (swapped == 0)
            break;
    }
    double sum = 0;
    for (int i = 0; i < n; i++ ) {
        double cost = (double)i - arr[i];
        if (cost<0) {
            cost = -cost;
        }
        sum += cost;
    }
    return sum;
}
double dsquare(double* ii, double* shadow) {
    // This returns the derivative of square or 2 * x
    return __enzyme_autodiff((void*) square, ii, shadow, 10);
}
int main() {
    double* ii = (double*)malloc(10*8);
    double* shadow = (double*)malloc(10*8);
    for(double i=1; i<5; i++) {
        for(double j=0; j<10; j++) {
            ii[(int)j] = 10 - j;
            shadow[(int)j] = 0;
        }
        dsquare(ii, shadow);
        for(double j=0; j<10; j++) {
            ii[(int)j] = 10 - j;
        }
        double tmp = square(ii, 10);
        for(int j = 0; j<10; j++) {
            printf("square(%f)=%f, dsquare(%f)=%f\n", i, tmp, i, shadow[j]);
        }
    }
}