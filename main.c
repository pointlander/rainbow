#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mnist.h"

// S is the softmax factor
const double S = (1.0 - 1e-300);
// B1 exponential decay of the rate for the first moment estimates
const double B1 = 0.8;
// B2 exponential decay rate for the second-moment estimates
const double B2 = 0.89;
// Eta is the learning rate
const double Eta = .1;

struct Slice {
    double* a;
    int size;
};

struct Slice MakeSlice(int size) {
    double *a = (double *)calloc(size, sizeof(double));
    struct Slice x = {
        .a = a,
        .size = size
    };
    return x;
}

void FreeSlice(struct Slice x) {
    free(x.a);
}

void Within(struct Slice a, int n, ...) {
    va_list ptr;
    va_start(ptr, n);
    for (int i = 0; i < n; i++) {
        int index = va_arg(ptr, int);
        if ((index < 0) || (index > a.size)) {
             printf("index out of bounds for slice\n");
            exit(1);
        }
    }
    va_end(ptr);
}

struct Slice Slice(struct Slice a, int begin, int end) {
    if ((end < begin) || (begin < 0) || (end > a.size)) {
        printf("index out of bounds for slice\n");
        exit(1);
    }
    struct Slice x = {
        .a = a.a+begin,
        .size = end - begin
    };
    return x;
}

struct Set {
    struct Slice T[3];
    struct Slice M[3];
    struct Slice V[3];
};

struct Set NewSet(int size) {
    struct Set set;
    for (int i = 0; i < 3; i++) {
        set.T[i] = MakeSlice(size);
        set.M[i] = MakeSlice(size);
        set.V[i] = MakeSlice(size);
    }
    return set;
}

void FreeSet(struct Set set) {
    for (int i = 0; i < 3; i++) {
        FreeSlice(set.T[i]);
        FreeSlice(set.M[i]);
        FreeSlice(set.V[i]);
    }
}

struct Data {
    int width;
    int rows;
    struct Slice images;
    char* labels;
    struct Slice entropy;
};

struct Data NewData(int width) {
    int rows = (NUM_TRAIN+NUM_TEST);
    struct Slice images = MakeSlice(rows*width);
    char* labels = (char*)malloc(rows);
    struct Slice entropy = MakeSlice(rows);
    struct Data data = {
        .width = width,
        .rows = rows,
        .images = images,
        .labels = labels,
        .entropy = entropy
    };
    int index = 0;
    Within(images, 1, NUM_TRAIN*width);
    for (int i = 0; i < NUM_TRAIN; i++) {
        double sum = 0;
        for (int j = 0; j < width; j++) {
            sum += (double)train_image_char[i][j];
        }
        for (int j = 0; j < width; j++) {
            images.a[index] = ((double)train_image_char[i][j])/sum;
            index++;
        }
        labels[i] = train_label_char[i][0];
        entropy.a[i] = 0;
    }
    Within(images, 2, NUM_TRAIN*width, (NUM_TRAIN+NUM_TEST)*width);
    for (int i = 0; i < NUM_TEST; i++) {
        double sum = 0;
        for (int j = 0; j < width; j++) {
            sum += (double)test_image_char[i][j];
        }
        for (int j = 0; j < width; j++) {
            images.a[index] = ((double)test_image_char[i][j])/sum;
            index++;
        }
        labels[NUM_TRAIN+i] = train_label_char[i][0];
        entropy.a[NUM_TRAIN+i] = 0;
    }
    return data;
}

struct Data NewZeroData(int width, int rows) {
    struct Slice images = MakeSlice(rows*width);
    char* labels = (char*)malloc(rows);
    struct Slice entropy = MakeSlice(rows);
    for (int i = 0; i < rows; i++) {
        labels[i] = 0;
    }
    struct Data data = {
        .width = width,
        .rows = rows,
        .images = images,
        .labels = labels,
        .entropy = entropy
    };
    return data;
}

void swap(struct Data data, int a, int b) {
    Within(data.images, 2, a*data.width + data.width, b*data.width + data.width);
    for (int k = 0; k < data.width; k++) {
        double s = data.images.a[a*data.width + k];
        data.images.a[a*data.width + k] = data.images.a[b*data.width + k];
        data.images.a[b*data.width + k] = s;
    }
    char c = data.labels[a];
    data.labels[a] = data.labels[b];
    data.labels[b] = c;
    double s = data.entropy.a[a];
    data.entropy.a[a] = data.entropy.a[b];
    data.entropy.a[b] = s;
}

int partition(struct Data data, int low, int high) {
    double pivot = data.entropy.a[low];
    int i = low;
    int j = high;

    Within(data.entropy, 1, j);
    while (i < j) {
        while (data.entropy.a[i] <= pivot && i <= high - 1) {
            i++;
        }
        while (data.entropy.a[j] > pivot && j >= low + 1) {
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
    quickSort(data, 0, data.rows - 1);
}

int IsSorted(struct Data data) {
    Within(data.entropy, 1, data.rows - 1);
    for (int i = 0; i < (data.rows - 1); i++) {
        if (data.entropy.a[i] > data.entropy.a[i+1]) {
            return 0;
        }
    }
    return 1;
}

void FreeData(struct Data data) {
    FreeSlice(data.images);
    free(data.labels);
    FreeSlice(data.entropy);
}

double dot(struct Slice x, struct Slice y) {
    double sum = 0;
    for (int i = 0; i < x.size; i++) {
        sum += x.a[i]*y.a[i];
    }
    return sum;
}

double dotT(struct Slice x, double* y, int col, int width) {
    double sum = 0;
    for (int i = 0; i < x.size; i++) {
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
    for (int i = 0; i < x.size; i++) {
        x.a[i] = exp(x.a[i] - s);
        sum += x.a[i];
    }
    for (int i = 0; i < x.size; i++) {
        x.a[i] /= sum;
    }
}

void SelfEntropy(struct Slice q, struct Slice k, struct Slice v, struct Slice e, int width) {
    const int cols = width;
    const int rows = q.size/width;
    struct Slice entropies = MakeSlice(cols);
    struct Slice values = MakeSlice(rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            values.a[j] = dot(Slice(q, i*width, (i+1)*width),
                Slice(k, j*width, (j+1)*width));
        }
        softmax(values);

        for (int j = 0; j < cols; j++) {
            entropies.a[j] = dotT(values, v.a, j, width);
        }
        softmax(entropies);

        double entropy = 0;
        Within(entropies, 1, entropies.size);
        for (int j = 0; j < entropies.size; j++) {
            entropy += entropies.a[j] * log(entropies.a[j]);
        }
        e.a[i] = -entropy;
    }
    FreeSlice(entropies);
    FreeSlice(values);
}

struct Data Transform(struct Data *data, struct Slice *t) {
    const int rows = t->size/data->width;
    struct Data cp = {
        .width = rows,
        .rows = data->rows,
        .images = MakeSlice(data->rows*rows),
        .labels = data->labels,
        .entropy = data->entropy
    };
    const int width = data->width;
    int index = 0;
    for (int i = 0; i < data->rows; i++) {
        for (int j = 0; j < rows; j++) {
            cp.images.a[index] = dot(Slice(*t, j*width, (j+1)*width),
                Slice(data->images, i*width, (i+1)*width));
            index++;
        }
    }
    return cp;
}

extern double __enzyme_autodiff(void*, struct Set*, struct Set*, struct Data*, struct Data*, double*, double*);
double rainbow(struct Set *set, struct Data *data, double *loss) {
    struct Data q = Transform(data, &(set->T[0]));
    struct Data k = Transform(data, &(set->T[1]));
    struct Data v = Transform(data, &(set->T[2]));
    SelfEntropy(q.images, k.images, v.images, data->entropy, q.width);
    double sum = 0;
    Within(data->entropy, 1, 100);
    for (int i = 0; i < 100; i++) {
        sum += data->entropy.a[i];
    }
    FreeSlice(q.images);
    FreeSlice(k.images);
    FreeSlice(v.images);
    *loss = sum;
    return sum;
}

double Pow(double x, int i) {
    return pow(x, (double)(i+1));
}

int main() {
    srand(1);
    load_mnist();
    struct Data data = NewData(SIZE);
    struct Data cp = NewZeroData(SIZE, 100);
    struct Set set = NewSet(SIZE*32);
    double factor = sqrt(2.0 / ((double)SIZE));
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < set.T[s].size; i++) {
            set.T[s].a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
        }
    }
    for (int e = 0; e < 100; e++) {
        struct Set d = NewSet(SIZE*32);
        double cost = 0;
        for (int i = 0; i < 2; i++) {
            printf("calculating self entropy\n");
            Within(cp.images, 1, 99*SIZE + SIZE);
            Within(data.images, 1, (data.rows - 1)*SIZE + SIZE);
            Within(cp.entropy, 1, 100);
            Within(data.entropy, 1, data.rows);
            for (int j = 0; j <= (data.rows - 100); j += 100) {
                for (int k = 0; k < 100; k++) {
                    for (int l = 0; l < SIZE; l++) {
                        cp.images.a[k*SIZE + l] = data.images.a[(k+j)*SIZE + l];
                    }
                    cp.labels[k] = data.labels[k + j];
                    cp.entropy.a[k] = data.entropy.a[k + j];
                }
                struct Data d_data = NewZeroData(SIZE, 100);
                double loss = 0;
                double dloss = 0;
                __enzyme_autodiff((void*) rainbow, &set, &d, &cp, &d_data, &loss, &dloss);
                cost += loss;
                FreeData(d_data);
                for (int k = 0; k < 100; k++) {
                    for (int l = 0; l < SIZE; l++) {
                        data.images.a[(k+j)*SIZE + l] = cp.images.a[k*SIZE + l];
                    }
                    data.labels[k + j] = cp.labels[k];
                    data.entropy.a[k + j] = cp.entropy.a[k];
                }
            }
            if (IsSorted(data)) {
                printf("is sorted\n");
                break;
            }
            printf("sorting\n");
            SortData(data);
            printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[(NUM_TRAIN+NUM_TEST)-1]);
        }
        double norm = 0;
        for (int s = 0; s < 3; s++) {
            Within(d.T[s], 1, d.T[s].size);
            for (int i = 0; i < d.T[s].size; i++) {
                norm += d.T[s].a[i] * d.T[s].a[i];
            }
        }
        norm = sqrt(norm);
        double scaling = 1;
        if (norm > 1) {
            scaling /= norm;
        }
        double b1 = Pow(B1, e);
        double b2 = Pow(B2, e);
        for (int s = 0; s < 3; s++) {
            Within(d.T[s], 1, d.T[s].size);
            Within(set.M[s], 1, d.T[s].size);
            Within(set.V[s], 1, d.T[s].size);
            for (int i = 0; i < d.T[s].size; i++) {
                double g = d.T[s].a[i] * scaling;
                double mm = B1*set.M[s].a[i] + (1-B1)*g;
                double vv = B2*set.V[s].a[i] + (1-B2)*g*g;
                set.M[s].a[i] = mm;
                set.V[s].a[i] = vv;
                double mhat = mm / (1 - b1);
                double vhat = vv / (1 - b2);
                if (vhat < 0) {
                    vhat = 0;
                }
                set.T[s].a[i] -= Eta * mhat / (sqrt(vhat) + 1e-8);
            }
        }
        FreeSet(d);
        printf("cost %f\n", cost);
    }
    printf("\n");
    FreeData(data);
    FreeData(cp);
    FreeSet(set);
}