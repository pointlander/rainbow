// test.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

#define S 1.0 - 1e-300

struct Slice {
    double* a;
    int size;
};

struct Slice NewSlice(double* a, int begin, int end) {
    struct Slice x = {a+begin, end};
    return x;
}

struct Data {
    double* images;
    char* labels;
    double* entropy;
};

struct Data NewData() {
    double* images = (double*)malloc((NUM_TRAIN+NUM_TEST)*SIZE*sizeof(double));
    char* labels = (char*)malloc((NUM_TRAIN+NUM_TEST));
    double* entropy = (double*)malloc((NUM_TRAIN+NUM_TEST)*sizeof(double));
    struct Data data = {images, labels, entropy};
    int index = 0;
    for (int i=0; i<NUM_TRAIN; i++) {
        double sum = 0;
        for (int j=0; j<SIZE; j++) {
            sum += (double)train_image_char[i][j];
        }
        for (int j=0; j<SIZE; j++) {
            images[index] = ((double)train_image_char[i][j])/sum;
            index++;
        }
        labels[i] = train_label_char[i][0];
        entropy[i] = 0;
    }
    for (int i=0; i<NUM_TEST; i++) {
        double sum = 0;
        for (int j=0; j<SIZE; j++) {
            sum += (double)test_image_char[i][j];
        }
        for (int j=0; j<SIZE; j++) {
            images[index] = ((double)test_image_char[i][j])/sum;
            index++;
        }
        labels[NUM_TRAIN+i] = train_label_char[i][0];
        entropy[NUM_TRAIN+i] = 0;
    }
    return data;
}

void SortData(struct Data data) {
    int swapped = 0;
    for (int i = 0; i < (NUM_TRAIN+NUM_TEST) - 1; i++) {
        swapped = 0;
        for (int j = 0; j < (NUM_TRAIN+NUM_TEST) - i - 1; j++) {
            if (data.entropy[j] > data.entropy[j + 1]) {
                for (int k = 0; k < SIZE; k++) {
                    double s = data.images[j*SIZE + k];
                    data.images[j*SIZE + k] = data.images[(j+1)*SIZE + k];
                    data.images[(j+1)*SIZE + k] = s;
                }
                char c = data.labels[j];
                data.labels[j] = data.labels[j + 1];
                data.labels[j + 1] = c;
                double s = data.entropy[j];
                data.entropy[j] = data.entropy[j + 1];
                data.entropy[j + 1] = s;
                swapped = 1;
            }
        }
 
        if (swapped == 0)
            break;
    }
}

int IsSorted(struct Data data) {
    for (int i = 0; i < (NUM_TRAIN+NUM_TEST) - 1; i++) {
        if (data.entropy[i] > data.entropy[i+1]) {
            return 0;
        }
    }
    return 1;
}

void DestroyData(struct Data data) {
    free(data.images);
    free(data.labels);
    free(data.entropy);
}

double dot(struct Slice x, struct Slice y) {
    double sum = 0;
    for (int i=0; i<x.size; i++) {
        sum += x.a[i]*y.a[i];
    }
    return sum;
}

double dotT(struct Slice x, double* y, int col) {
    double sum = 0;
    for (int i=0; i<x.size; i++) {
        sum += x.a[i]*y[i*SIZE+col];
    }
    return sum;
}

void softmax(struct Slice x) {
    double max = 0;
    for (int i=0; i<x.size; i++) {
        if (x.a[i] > max) {
            max = x.a[i];
        }
    }
    double s = max*S;
    double sum = 0;
    for (int i=0; i<x.size; i++) {
        x.a[i] = exp(x.a[i] - s);
        sum += x.a[i];
    }
    for (int i=0; i<x.size; i++) {
        x.a[i] /= sum;
    }
}

void SelfEntropy(struct Slice images, struct Slice e) {
    int cols = SIZE;
    int rows = images.size/SIZE;
    struct Slice entropies = {(double*)malloc(cols*sizeof(double)), cols};
    struct Slice values = {(double*)malloc(cols*sizeof(double)), rows};
    for (int i=0; i<rows; i++) {
        for (int j=0; j<rows; j++) {
            values.a[j] = dot(NewSlice(images.a, i*SIZE, (i+1)*SIZE), NewSlice(images.a, j*SIZE, (j+1)*SIZE));
        }
        softmax(values);

        for (int j=0; j<cols; j++) {
            entropies.a[j] = dotT(values, images.a, j);
        }
        softmax(entropies);

        double entropy = 0;
        for (int j=0; j<entropies.size; j++) {
            entropy += entropies.a[j] * log(entropies.a[j]);
        }
        e.a[i] = -entropy;
    }
    free(entropies.a);
    free(values.a);
}

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
    load_mnist();
    struct Data data = NewData();
    SelfEntropy(NewSlice(data.images, 0, 100*SIZE), NewSlice(data.entropy, 0, 100));
    SortData(data);
    printf("%d\n", IsSorted(data));
    printf("%f\n", data.entropy[0]);

    double* ii = (double*)malloc(10*sizeof(double));
    double* shadow = (double*)malloc(10*sizeof(double));
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

    DestroyData(data);
    free(ii);
    free(shadow);
}