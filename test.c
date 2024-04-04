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
    int rows;
    double* images;
    char* labels;
    double* entropy;
};

struct Data NewData(int width) {
    int rows = (NUM_TRAIN+NUM_TEST);
    double* images = (double*)malloc(rows*width*sizeof(double));
    char* labels = (char*)malloc(rows);
    double* entropy = (double*)malloc(rows*sizeof(double));
    struct Data data = {width, rows, images, labels, entropy};
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

struct Data NewZeroData(int width, int rows) {
    double *images = malloc(rows*width*sizeof(double));
    for (int i = 0; i < rows*width; i++) {
        images[i] = 0;
    }
    char* labels = (char*)malloc(rows);
    double* entropy = (double*)malloc(rows*sizeof(double));
    for (int i = 0; i < rows; i++) {
        labels[i] = 0;
        entropy[i] = 0;
    }
    struct Data data = {width, rows, images, labels, entropy};
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

inline double dot(struct Slice x, struct Slice y) {
    double sum = 0;
    for (int i=0; i<x.size; i++) {
        sum += x.a[i]*y.a[i];
    }
    return sum;
}

inline double dotT(struct Slice x, double* y, int col, int width) {
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
    const int cols = width;
    const int rows = images.size/width;
    struct Slice entropies = {(double*)malloc(cols*sizeof(double)), cols};
    struct Slice values = {(double*)malloc(rows*sizeof(double)), rows};
    for (int i=0; i<rows; i++) {
        for (int j=0; j<rows; j++) {
            values.a[j] = dot(NewSlice(images.a, i*width, (i+1)*width),
                NewSlice(images.a, j*width, (j+1)*width));
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

struct Data Transform(struct Data *data, struct Slice *t) {
    const int rows = t->size/data->width;
    struct Data cp = {rows, data->rows, malloc(data->rows*rows*sizeof(double)), data->labels, data->entropy};
    const int width = data->width;
    double *a = t->a;
    double *b = data->images;
    int index = 0;
    for (int i = 0; i < data->rows; i++) {
        for (int j = 0; j < rows; j++) {
            cp.images[index] = dot(NewSlice(a, j*width, (j+1)*width),
                NewSlice(b, i*width, (i+1)*width));
            index++;
        }
    }
    return cp;
}

void Rainbow(struct Data *data) {
    struct Slice a = NewSlice(data->images, 0, 100*data->width);
    struct Slice b = NewSlice(data->entropy, 0, 100);
    SelfEntropy(a, b, data->width);
}

extern double __enzyme_autodiff(void*, struct Slice*, struct Slice*, struct Data*, struct Data*);
double rainbow(struct Slice *t, struct Data *data) {
    struct Data dat = Transform(data, t);
    Rainbow(&dat);
    double sum = 0;
    for (int i = 0; i < 100; i++) {
        data->entropy[i] = dat.entropy[i];
        sum += dat.entropy[i];
    }
    free(dat.images);
    return sum;
}

int main() {
    srand(1);
    load_mnist();
    struct Data data = NewData(SIZE);
    struct Data cp = NewZeroData(SIZE, 100);
    struct Data d_data = NewZeroData(SIZE, 100);
    struct Slice t = {malloc(SIZE*32*sizeof(double)), SIZE*32};
    struct Slice d = {malloc(SIZE*32*sizeof(double)), SIZE*32};
    double factor = sqrt(2.0 / ((double)SIZE));
    for (int i = 0; i < t.size; i++) {
        t.a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
        d.a[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
        printf("calculating self entropy\n");
        for (int j = 0; j <= (data.rows - 100); j += 100) {
            for (int k = 0; k < 100; k++) {
                for (int l = 0; l < SIZE; l++) {
                    cp.images[k*SIZE + l] = data.images[(k+j)*SIZE + l];
                }
                cp.labels[k] = data.labels[k + j];
                cp.entropy[k] = data.entropy[k + j];
            }
            __enzyme_autodiff((void*) rainbow, &t, &d, &cp, &d_data);
            for (int k = 0; k < 100; k++) {
                for (int l = 0; l < SIZE; l++) {
                    data.images[(k+j)*SIZE + l] = cp.images[k*SIZE + l];
                }
                data.labels[k + j] = cp.labels[k];
                data.entropy[k + j] = cp.entropy[k];
            }
        }
        if (IsSorted(data)) {
            break;
        }
        printf("sorting\n");
        SortData(data);
        printf("%.17f %.17f\n", data.entropy[0], data.entropy[(NUM_TRAIN+NUM_TEST)-1]);
    }
    for (int i = 0; i < SIZE*32; i++) {
        printf("%.20f ", d.a[i]);
    }
    printf("\n");
    DestroyData(data);
    DestroyData(cp);
    DestroyData(d_data);
}