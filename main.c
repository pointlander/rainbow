#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "mnist/mnist.h"
#include "set.pb-c.h"

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

void Within(struct Slice a, int n) {
    if ((n < 0) || (n > a.size)) {
        printf("index out of bounds for slice\n");
        exit(1);
    }
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
    Within(images, NUM_TRAIN*width);
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
    Within(images, NUM_TRAIN*width);
    Within(images, (NUM_TRAIN+NUM_TEST)*width);
    for (int i = 0; i < NUM_TEST; i++) {
        double sum = 0;
        for (int j = 0; j < width; j++) {
            sum += (double)test_image_char[i][j];
        }
        for (int j = 0; j < width; j++) {
            images.a[index] = ((double)test_image_char[i][j])/sum;
            index++;
        }
        labels[NUM_TRAIN+i] = test_label_char[i][0] + 10;
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
    Within(data.images, a*data.width + data.width);
    Within(data.images, b*data.width + data.width);
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

    Within(data.entropy, j);
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
    Within(data.entropy, data.rows - 1);
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

struct Slice SelfEntropy(struct Slice q, struct Slice k, struct Slice v, int width) {
    const int cols = width;
    const int rows = q.size/width;
    struct Slice entropies = MakeSlice(cols);
    struct Slice values = MakeSlice(rows);
    struct Slice e = MakeSlice(rows);
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
        Within(entropies, entropies.size);
        for (int j = 0; j < entropies.size; j++) {
            entropy += entropies.a[j] * log(entropies.a[j]);
        }
        e.a[i] = -entropy;
    }
    FreeSlice(entropies);
    FreeSlice(values);
    return e;
}

struct Slice Transform(struct Data *data, struct Slice *t) {
    const int rows = t->size/data->width;
    struct Slice images = MakeSlice(data->rows*rows);
    const int width = data->width;
    int index = 0;
    for (int i = 0; i < data->rows; i++) {
        for (int j = 0; j < rows; j++) {
            images.a[index] = dot(Slice(*t, j*width, (j+1)*width),
                Slice(data->images, i*width, (i+1)*width));
            index++;
        }
    }
    return images;
}

extern double __enzyme_autodiff(void*, struct Set*, struct Set*, struct Data*, struct Data*, double*, double*);
double rainbow(struct Set *set, struct Data *data, double *loss) {
    struct Slice q = Transform(data, &(set->T[0]));
    struct Slice k = Transform(data, &(set->T[1]));
    struct Slice v = Transform(data, &(set->T[2]));
    struct Slice e = SelfEntropy(q, k, v, (set->T[0].size)/data->width);
    FreeSlice(q);
    FreeSlice(k);
    FreeSlice(v);
    Within(e, 100);
    double sum = 0;
    for (int i = 0; i < e.size; i++) {
        data->entropy.a[i] = e.a[i];
        sum += e.a[i];
    }
    FreeSlice(e);
    *loss = sum;
    return sum;
}

double Pow(double x, int i) {
    return pow(x, (double)(i+1));
}

struct Thread {
    int offset;
    int batchSize;
    pthread_t thread;
    struct Data data;
    struct Set set;
    struct Set d;
    double cost;
};

void *Rainbow(void *ptr) {
    struct Thread *t = (struct Thread*)ptr;
    struct Set d = NewSet(SIZE*32);
    t->d = d; 
    struct Data cp = NewZeroData(SIZE, 100);
    Within(cp.images, 99*SIZE + SIZE);
    Within(cp.entropy, 100);
    for (int j = t->offset; j < (t->offset + t->batchSize); j++) {
        for (int k = 0; k < 100; k++) {
            for (int l = 0; l < SIZE; l++) {
                cp.images.a[k*SIZE + l] = t->data.images.a[(k+j*100)*SIZE + l];
            }
            cp.labels[k] = t->data.labels[k + j*100];
            cp.entropy.a[k] = t->data.entropy.a[k + j*100];
        }
        struct Data d_data = NewZeroData(SIZE, 100);
        double loss = 0;
        double dloss = 0;
        __enzyme_autodiff((void*) rainbow, &(t->set), &d, &cp, &d_data, &loss, &dloss);
        t->cost += loss;
        FreeData(d_data);
        for (int k = 0; k < 100; k++) {
            for (int l = 0; l < SIZE; l++) {
                t->data.images.a[(k+j*100)*SIZE + l] = cp.images.a[k*SIZE + l];
            }
            t->data.labels[k + j*100] = cp.labels[k];
            t->data.entropy.a[k + j*100] = cp.entropy.a[k];
        }
    }
    FreeData(cp);
    return 0;
}

FILE *fp = NULL;

void handler(int sig) {
    fclose(fp);
    exit(0);
}

int main(int argc, char *argv[]) {
    srand(1);
    signal(SIGINT, handler);
    load_mnist();

    if (argc > 1) {
        printf("weights %s\n", argv[1]);
        FILE *f = fopen(argv[1], "rb");
        if (f == NULL) {
            printf("Error opening file!\n");
            return 1;
        }
        int result = fseek(f, 0, SEEK_END);
        if (result == EOF) {
            printf("Error seeking in file!\n");
            fclose(f);
            return 1;
        }
        long fsize = ftell(f);
        result = fseek(f, 0, SEEK_SET);
        if (result == EOF) {
            printf("Error seeking in file!\n");
            fclose(f);
            return 1;
        }
        uint8_t *buf = malloc(fsize);
        result = fread(buf, fsize, 1, f);
        if (result == EOF) {
            printf("Error reading from file!\n");
            fclose(f);
            return 1;
        }
        result = fclose(f);
        if (result == EOF) {
            printf("Error closing file!\n");
            fclose(f);
            return 1;
        }
        struct ProtoTf64__Set *set;
        set = proto_tf64__set__unpack(NULL, fsize, buf);
        struct Set weights;
        for (int i = 0; i < 3; i++) {
            weights.T[i].size = set->weights[i]->n_values;
            weights.T[i].a = set->weights[i]->values;
            weights.M[i].size = set->weights[i]->n_values;
            weights.M[i].a = set->weights[i]->states;
            weights.V[i].size = set->weights[i]->n_values;
            weights.V[i].a = set->weights[i]->states + set->weights[i]->n_values;
        }
        struct Data data = NewData(SIZE);

        for (int i = 0; i < 16; i++) {
            printf("calculating self entropy\n");
            struct Data cp = NewZeroData(SIZE, 100);
            Within(cp.images, 99*SIZE + SIZE);
            Within(cp.entropy, 100);
            for (int j = 0; j < data.rows; j += 100) {
                for (int k = 0; k < 100; k++) {
                    for (int l = 0; l < SIZE; l++) {
                        cp.images.a[k*SIZE + l] = data.images.a[(k+j)*SIZE + l];
                    }
                    cp.labels[k] = data.labels[k + j];
                    cp.entropy.a[k] = data.entropy.a[k + j];
                }
                double loss = 0;
                rainbow(&weights, &cp, &loss);
                for (int k = 0; k < 100; k++) {
                    for (int l = 0; l < SIZE; l++) {
                        data.images.a[(k+j)*SIZE + l] = cp.images.a[k*SIZE + l];
                    }
                    data.labels[k + j] = cp.labels[k];
                    data.entropy.a[k + j] = cp.entropy.a[k];
                }
            }
            FreeData(cp);
            if (IsSorted(data)) {
                printf("is sorted\n");
                printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[(NUM_TRAIN+NUM_TEST)-1]);
                break;
            }
            printf("sorting\n");
            SortData(data);
            printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[(NUM_TRAIN+NUM_TEST)-1]);
        }

        int correct = 0;
        for (int i = 0; i < (data.rows-1); i++) {
            char current = data.labels[i];
            if (current > 9) {
                current -= 10;
                char next = data.labels[i+1];
                if (next > 9) {
                    next -= 10;
                }
                if (current == next) {
                    correct++;
                }
            }
        }
        printf("correct %d %f\n", correct, ((double)correct)/((double)10000));

        FreeData(data);
        proto_tf64__set__free_unpacked(set, NULL);
        free(buf);
        printf("\n");
        exit(1);
    }

    fp = fopen("epochs.txt", "w");
    if (fp == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    int result = fprintf(fp, "# epoch cost\n");
    if (result == EOF) {
        printf("Error writing to file!\n");
        fclose(fp);
        return 1;
    }
    const int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    struct Data data = NewData(SIZE);
    const int batchSize = (data.rows/100)/numCPU;
    const int spares = (data.rows/100)%numCPU;
    printf("%d %d %d\n", numCPU, batchSize, spares);
    struct Set set = NewSet(SIZE*32);
    double factor = sqrt(2.0 / ((double)SIZE));
    for (int s = 0; s < 3; s++) {
        for (int i = 0; i < set.T[s].size; i++) {
            set.T[s].a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
        }
    }
    double cost = 0;
    const int epochs = 100;
    for (int e = 0; e < epochs; e++) {
        struct Set d = NewSet(SIZE*32);
        cost = 0;
        for (int i = 0; i < 2; i++) {
            printf("calculating self entropy\n");
            Within(data.images, (data.rows - 1)*SIZE + SIZE);
            Within(data.entropy, data.rows);
            struct Thread threads[numCPU];
            int j = 0;
            int cpu = 0;
            while (j < (numCPU*batchSize + spares)) {
                threads[cpu].offset = j;
                threads[cpu].batchSize = batchSize;
                if (cpu == 0) {
                    threads[cpu].batchSize += spares;
                }
                threads[cpu].data = data;
                threads[cpu].set = set;
                threads[cpu].cost = 0;
                pthread_create(&threads[cpu].thread, 0, Rainbow, (void*)&threads[cpu]);
                j += threads[cpu].batchSize;
                cpu++;
            }
            for (int j = 0; j < numCPU; j++) {
                pthread_join(threads[j].thread, NULL);
                for (int l = 0; l < 3; l++) {
                    for (int k = 0; k < d.T[l].size; k++) {
                        d.T[l].a[k] += threads[j].d.T[l].a[k];
                    }
                }
                cost += threads[j].cost;
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
            Within(d.T[s], d.T[s].size);
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
            Within(d.T[s], d.T[s].size);
            Within(set.M[s], d.T[s].size);
            Within(set.V[s], d.T[s].size);
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
        int result = fprintf(fp, "%d %f\n", e, cost);
        if (result == EOF) {
            printf("Error writing to file!\n");
            fclose(fp);
            return 1;
        }
    }
    printf("\n");
    struct ProtoTf64__Set protoSet = PROTO_TF64__SET__INIT;
    struct ProtoTf64__Weights *weights[3];
    for (int i = 0; i < 3; i++) {
        struct ProtoTf64__Weights *weight = calloc(1, sizeof(struct ProtoTf64__Weights));
        *weight = (struct ProtoTf64__Weights)PROTO_TF64__WEIGHTS__INIT;
        weight->n_shape = 2;
        weight->shape = calloc(2, sizeof(int64_t));
        weight->shape[0] = SIZE;
        weight->shape[1] = 32;
        weight->n_values = set.T[i].size;
        weight->values = set.T[i].a;
        weight->n_states = set.M[i].size + set.V[i].size;
        weight->states = calloc(weight->n_states, sizeof(double));
        int index = 0;
        for (int j = 0; j < set.M[i].size; j++) {
            weight->states[index] = set.M[i].a[index];
            index++;
        }
        for (int j = 0; j < set.V[i].size; j++) {
            weight->states[index] = set.V[i].a[index];
            index++;
        }
        weights[i] = weight;
    }
    protoSet.cost = cost;
    protoSet.epoch = epochs;
    protoSet.n_weights = 3;
    protoSet.weights = weights;
    void *buf;
    unsigned len;
    len = proto_tf64__set__get_packed_size(&protoSet);
    buf = malloc(len);
    proto_tf64__set__pack(&protoSet, buf);
    FILE *output = fopen("weights.bin", "w");
    if (fp == NULL) {
        printf("Error opening weights file!\n");
        return 1;
    };
    result = fwrite(buf,len,1,output);
    if (result == EOF) {
        printf("Error writing to file!\n");
        fclose(fp);
        return 1;
    }
    fclose(output);
    fclose(fp);
    FreeData(data);
    FreeSet(set);
}