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

inline double dot(double *x, int xbegin, int xend, double *y, int ybegin, int yend) {
    double sum = 0;
    int size = xend - xbegin;
    for (int i=0; i<size; i++) {
        sum += x[i+xbegin]*y[i+ybegin];
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
    int cols = SIZE;
    int rows = images.size/width;
    struct Slice entropies = {(double*)malloc(cols*sizeof(double)), cols};
    struct Slice values = {(double*)malloc(rows*sizeof(double)), rows};
    for (int i=0; i<rows; i++) {
        for (int j=0; j<rows; j++) {
            //values.a[j] = dot(NewSlice(images.a, i*width, (i+1)*width), NewSlice(images.a, j*width, (j+1)*width));
            values.a[j] = dot(images.a, i*width, (i+1)*width,
                images.a, j*width, (j+1)*width);
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
    printf("in transform a\n");
    int rows = t->size/data->width;
    printf("in transform b\n");
    struct Data cp = {rows, data->rows, malloc(data->rows*rows*sizeof(double)), data->labels, data->entropy};
    printf("in transform c\n");
    printf("%d %d\n", data->rows, rows);
    const int width = data->width;
    const double *a = t->a;
    const double *b = data->images;
    int index = 0;
    for (int i = 0; i < data->rows; i++) {
        const int yoffset = i*width;
        for (int j = 0; j < rows; j++) {
            const int xoffset = j*width;
            printf("%d %d\n", i, j);
            //struct Slice a = NewSlice(t->a, j*data->width, (j+1)*data->width);
            //struct Slice b = NewSlice(data->images, i*data->width, (i+1)*data->width);
            //cp.images[i*rows+j] = dot(t->a, j*data->width, (j+1)*data->width, 
            //    data->images, i*data->width, (i+1)*data->width);
            double sum = 0;
            int xindex = xoffset;
            int yindex = yoffset;
            for (int k = 0; k < width; k++) {
                sum += a[xindex]*b[yindex];
                xindex++;
                yindex++;
            }
            cp.images[index] = sum;
            index++;
        }
    }
    return cp;
}

void Rainbow(struct Data data, int iterations) {
    for (int i = 0; i < iterations; i++) {
        printf("%d/%d\n", i, iterations);
        for (int j = 0; j <= data.rows - 100; j += 100) {
            struct Slice a = NewSlice(data.images, j*data.width, (j+100)*data.width);
            struct Slice b = NewSlice(data.entropy, j, j+100);
            SelfEntropy(a, b, data.width);
        }
        if (IsSorted(data)) {
            break;
        }
        printf("sorting...\n");
        SortData(data);
        printf("%.17f %.17f\n", data.entropy[0], data.entropy[(NUM_TRAIN+NUM_TEST)-1]);
    }
}

extern double __enzyme_autodiff(void*, struct Slice*, struct Slice*, struct Data*, struct Data*);
double rainbow(struct Slice *t, struct Data *data) {
    printf("transform\n");
    struct Data dat = Transform(data, t);
    printf("rainbow\n");
    Rainbow(dat, 3);
    double sum = 0;
    for (int i = 0; i < (NUM_TRAIN+NUM_TEST); i++) {
        sum += dat.entropy[i];
    }
    //printf("sum %f\n", sum);
    return sum;
}
int main() {
    srand(1);
    load_mnist();
    struct Data data = NewData(SIZE);
    char* labels = (char*)malloc((NUM_TRAIN+NUM_TEST));
    double* entropy = (double*)malloc((NUM_TRAIN+NUM_TEST)*sizeof(double));
    for (int i = 0; i < (NUM_TRAIN+NUM_TEST); i++) {
        labels[i] = 0;
        entropy[i] = 0;
    }
    struct Data d_data = {32, (NUM_TRAIN+NUM_TEST), malloc((NUM_TRAIN+NUM_TEST)*32*sizeof(double)), labels, entropy};
    struct Slice t = {malloc(SIZE*32*sizeof(double)), SIZE*32};
    struct Slice d = {malloc(SIZE*32*sizeof(double)), SIZE*32};
    double factor = sqrt(2.0 / ((double)SIZE));
    for (int i = 0; i < t.size; i++) {
        t.a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
        d.a[i] = 0;
    }
    printf("autodiff\n");
    __enzyme_autodiff((void*) rainbow, &t, &d, &data, &d_data);
    for (int i = 0; i < t.size; i++) {
        printf("%f ", d.a[i]);
    }
    printf("\n");
    DestroyData(data);
}