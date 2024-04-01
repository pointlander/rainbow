// test.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

#define S (1.0 - 1e-300)

struct Slice {
    double* a;
    int size;
};

struct Slice NewSlice(double* a, int begin, int end) {
    struct Slice x = {a+begin, end - begin};
    return x;
}

struct Data {
    int width;
    double* images;
    char* labels;
    double* entropy;
};

struct Data NewData(int width) {
    double* images = (double*)malloc((NUM_TRAIN+NUM_TEST)*width*sizeof(double));
    char* labels = (char*)malloc((NUM_TRAIN+NUM_TEST));
    double* entropy = (double*)malloc((NUM_TRAIN+NUM_TEST)*sizeof(double));
    struct Data data = {width, images, labels, entropy};
    int index = 0;
    for (int i=0; i<NUM_TRAIN; i++) {
        double sum = 0;
        for (int j=0; j<width; j++) {
            sum += (double)train_image_char[i][j];
        }
        for (int j=0; j<width; j++) {
            images[index] = ((double)train_image_char[i][j])/sum;
            index++;
        }
        labels[i] = train_label_char[i][0];
        entropy[i] = 0;
    }
    for (int i=0; i<NUM_TEST; i++) {
        double sum = 0;
        for (int j=0; j<width; j++) {
            sum += (double)test_image_char[i][j];
        }
        for (int j=0; j<width; j++) {
            images[index] = ((double)test_image_char[i][j])/sum;
            index++;
        }
        labels[NUM_TRAIN+i] = train_label_char[i][0];
        entropy[NUM_TRAIN+i] = 0;
    }
    return data;
}

void swap(struct Data data, int a, int b) {
    for (int k = 0; k < data.width; k++) {
        double s = data.images[a*data.width + k];
        data.images[a*data.width + k] = data.images[b*data.width + k];
        data.images[b*data.width + k] = s;
    }
    char c = data.labels[a];
    data.labels[a] = data.labels[b];
    data.labels[b] = c;
    double s = data.entropy[a];
    data.entropy[a] = data.entropy[b];
    data.entropy[b] = s;
}

int partition(struct Data data, int low, int high) {
    double pivot = data.entropy[low];
    int i = low;
    int j = high;

    while (i < j) {
        while (data.entropy[i] <= pivot && i <= high - 1) {
            i++;
        }
        while (data.entropy[j] > pivot && j >= low + 1) {
            j--;
        }
        if (i < j) {
            swap(data, i, j);
        }
    }
    swap(data, low, j);
    return j;
}

void quickSort(struct Data data, int low, int high) {
    if (low < high) {
        int partitionIndex = partition(data, low, high);
        quickSort(data, low, partitionIndex - 1);
        quickSort(data, partitionIndex + 1, high);
    }
}

void SortData(struct Data data) {
    quickSort(data, 0, (NUM_TRAIN+NUM_TEST) - 1);
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

double dotT(struct Slice x, double* y, int col, int width) {
    double sum = 0;
    for (int i=0; i<x.size; i++) {
        sum += x.a[i]*y[i*width+col];
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

void SelfEntropy(struct Slice images, struct Slice e, int width) {
    int cols = SIZE;
    int rows = images.size/width;
    struct Slice entropies = {(double*)malloc(cols*sizeof(double)), cols};
    struct Slice values = {(double*)malloc(rows*sizeof(double)), rows};
    for (int i=0; i<rows; i++) {
        for (int j=0; j<rows; j++) {
            values.a[j] = dot(NewSlice(images.a, i*width, (i+1)*width), NewSlice(images.a, j*width, (j+1)*width));
        }
        softmax(values);

        for (int j=0; j<cols; j++) {
            entropies.a[j] = dotT(values, images.a, j, width);
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

void Rainbow(struct Data data, int iterations) {
    for (int i = 0; i < iterations; i++) {
        printf("%d/%d\n", i, iterations);
        for (int j = 0; j <= (NUM_TRAIN+NUM_TEST) - 100; j += 100) {
            SelfEntropy(NewSlice(data.images, j*data.width, (j+100)*data.width), NewSlice(data.entropy, j, j+100), data.width);
        }
        if (IsSorted(data)) {
            break;
        }
        printf("sorting...\n");
        SortData(data);
        printf("%f\n", data.entropy[0]);
    }
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
    struct Data data = NewData(SIZE);
    Rainbow(data, 64);

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