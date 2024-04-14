#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
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

char *Bible;
long BibleSize;
double Markov[256][256][256];

struct Slice {
    double* a;
    int size;
    int cols;
    int rows;
};

struct Slice MakeSlice(int size) {
    double *a = (double *)calloc(size, sizeof(double));
    struct Slice x = {
        .a = a,
        .size = size,
        .cols = 1,
        .rows = size,
    };
    return x;
}

struct Slice MakeMatrix(int cols, int rows) {
    const int size = cols * rows;
    double *a = (double *)calloc(size, sizeof(double));
    struct Slice x = {
        .a = a,
        .size = size,
        .cols = cols,
        .rows = rows
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
        .size = end - begin,
        .cols = 1,
        .rows = end - begin
    };
    return x;
}

struct Set {
    double loss;
    int cols;
    int rows;
    struct Slice T[4];
    struct Slice M[4];
    struct Slice V[4];
};

struct Set NewSet(int cols, int rows) {
    struct Set set;
    set.cols = cols;
    set.rows = rows;
    for (int i = 0; i < 3; i++) {
        set.T[i] = MakeMatrix(cols, rows);
        set.M[i] = MakeMatrix(cols, rows);
        set.V[i] = MakeMatrix(cols, rows);
    }
    set.T[3] = MakeMatrix(rows, 256);
    set.M[3] = MakeMatrix(rows, 256);
    set.V[3] = MakeMatrix(rows, 256);
    return set;
}

void FreeSet(struct Set set) {
    for (int i = 0; i < 4; i++) {
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
    struct Slice vectors;
    struct Slice entropy;
    int swaps;
};

struct Data NewData(int width) {
    const int rows = (NUM_TRAIN+NUM_TEST);
    struct Slice images = MakeSlice(rows*width);
    char* labels = (char*)calloc(rows, sizeof(char));
    struct Slice vectors = MakeMatrix(32, rows);
    struct Slice entropy = MakeSlice(rows);
    struct Data data = {
        .width = width,
        .rows = rows,
        .images = images,
        .labels = labels,
        .vectors = vectors,
        .entropy = entropy,
        .swaps = 0
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
    }
    Within(images, NUM_TRAIN*width);
    Within(images, rows*width);
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
    }
    return data;
}

struct Data NewBibleData(int offset) {
    const int width = 256;
    const int rows = 4000;
    struct Slice images = MakeSlice(rows*width);
    char* labels = (char*)calloc(rows, sizeof(char));
    struct Slice vectors = MakeMatrix(32, rows);
    struct Slice entropy = MakeSlice(rows);
    struct Data data = {
        .width = width,
        .rows = rows,
        .images = images,
        .labels = labels,
        .vectors = vectors,
        .entropy = entropy,
        .swaps = 0
    };
    int index = 0;
    Within(images, rows);
    char label = Bible[offset+rows];
    uint8_t last = 0;
    for (int i = 0; i < rows; i++) {
        const uint8_t current = Bible[offset+i];
        for (int j = 0; j < width; j++) {
            images.a[index] = Markov[last][current][j];
            index++;
        }
        labels[i] = label;
        last = current;
    }
    return data;
}

struct Data NewZeroData(int width, int rows) {
    struct Slice images = MakeSlice(rows*width);
    char* labels = (char*)calloc(rows, sizeof(char));
    struct Slice vectors = MakeMatrix(32, rows);
    struct Slice entropy = MakeSlice(rows);
    struct Data data = {
        .width = width,
        .rows = rows,
        .images = images,
        .labels = labels,
        .vectors = vectors,
        .entropy = entropy,
        .swaps = 0
    };
    return data;
}

void swap(struct Data *data, int a, int b) {
    const int width = data->width;
    Within(data->images, a*width + width);
    Within(data->images, b*width + width);
    for (int k = 0; k < width; k++) {
        double s = data->images.a[a*width + k];
        data->images.a[a*width + k] = data->images.a[b*width + k];
        data->images.a[b*width + k] = s;
    }
    const int cols = data->vectors.cols;
    for (int k = 0; k < cols; k++) {
        double s = data->vectors.a[a*cols + k];
        data->vectors.a[a*cols + k] = data->vectors.a[b*cols + k];
        data->vectors.a[b*cols + k] = s;
    }
    char c = data->labels[a];
    data->labels[a] = data->labels[b];
    data->labels[b] = c;
    double s = data->entropy.a[a];
    data->entropy.a[a] = data->entropy.a[b];
    data->entropy.a[b] = s;
}

int partition(struct Data *data, int low, int high) {
    double pivot = data->entropy.a[low];
    int i = low;
    int j = high;

    Within(data->entropy, j);
    while (i < j) {
        while (data->entropy.a[i] <= pivot && i <= high - 1) {
            i++;
        }
        while (data->entropy.a[j] > pivot && j >= low + 1) {
            j--;
        }
        if (i < j) {
            swap(data, i, j);
            data->swaps++;
        }
    }
    swap(data, low, j);
    data->swaps++;
    return j;
}

void quickSort(struct Data *data, int low, int high) {
    if (low < high) {
        int partitionIndex = partition(data, low, high);
        quickSort(data, low, partitionIndex - 1);
        quickSort(data, partitionIndex + 1, high);
    }
}

int SortData(struct Data *data) {
    data->swaps = 0;
    quickSort(data, 0, data->rows - 1);
    return data->swaps;
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
    FreeSlice(data.vectors);
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

void SelfEntropy(struct Data *data, struct Set *set) {
    struct Slice inputs[3] = {
        MakeMatrix(set->T[0].rows, data->rows),
        MakeMatrix(set->T[1].rows, data->rows),
        MakeMatrix(set->T[2].rows, data->rows)
    };
    for (int x = 0; x < 3; x++) {
        const int width = data->width;
        int index = 0;
        for (int i = 0; i < data->rows; i++) {
            struct Slice a = Slice(data->images, i*width, (i+1)*width);
            const int rows = set->T[x].rows;
            for (int j = 0; j < rows; j++) {
                struct Slice b = Slice(set->T[x], j*width, (j+1)*width);
                inputs[x].a[index] = dot(b, a);
                index++;
            }
        }
    }

    const int cols = inputs[0].cols;
    const int rows = inputs[0].rows;
    struct Slice entropies = MakeSlice(cols);
    struct Slice values = MakeSlice(rows);
    for (int i = 0; i < rows; i++) {
        struct Slice a = Slice(inputs[0], i*cols, (i+1)*cols);
        for (int j = 0; j < rows; j++) {
            struct Slice b = Slice(inputs[1], j*cols, (j+1)*cols);
            values.a[j] = dot(a, b);
        }
        softmax(values);

        for (int j = 0; j < cols; j++) {
            entropies.a[j] = dotT(values, inputs[2].a, j, cols);
            data->vectors.a[i*cols + j] = entropies.a[j];
        }
        softmax(entropies);

        double entropy = 0;
        Within(entropies, entropies.size);
        for (int j = 0; j < entropies.size; j++) {
            entropy += entropies.a[j] * log(entropies.a[j]);
        }
        data->entropy.a[i] = -entropy;
    }
    FreeSlice(entropies);
    FreeSlice(values);
    FreeSlice(inputs[0]);
    FreeSlice(inputs[1]);
    FreeSlice(inputs[2]);
}

extern double __enzyme_autodiff(void*, struct Set*, struct Set*, struct Data*, struct Data*);
double rainbow(struct Set *set, struct Data *data) {
    SelfEntropy(data, set);
    Within(data->entropy, 100);
    double sum = 0;
    for (int i = 0; i < data->entropy.size; i++) {
        sum += data->entropy.a[i];
    }
    set->loss = sum;
    return sum;
}
double rainbow_autodiff(struct Set *set, struct Set *d_set, struct Data *data, struct Data *d_data) {
    return __enzyme_autodiff((void*)rainbow, set, d_set, data, d_data);
}

extern double __enzyme_autodiffLang(void*, struct Set*, struct Set*, struct Data*, struct Data*);
void outputTransform(struct Set *set, struct Slice *in, struct Slice *out) {
    for (int j = 0; j < set->T[3].rows; j++) {
        struct Slice b = Slice(set->T[3], j*set->T[3].cols, (j + 1)*set->T[3].cols);
        out->a[j] = dot(*in, b);
    }
    softmax(*out);
}
double crossEntropy(int symbol, struct Slice *in) {
    double s = 0;
    for (int j = 0; j < in->size; j++) {
        if (j == symbol) {
            s += log(in->a[j] + .001);
        } else {
            s += log(1 - in->a[j] + .001);
        }
    }
    return -s;
}
double rainbowLang(struct Set *set, struct Data *data) {
    SelfEntropy(data, set);
    Within(data->vectors, 100 * 32);
    double sum = 0;
    struct Slice vector = MakeSlice(256);
    for (int i = 0; i < data->vectors.rows; i++) {
        struct Slice a = Slice(data->vectors, i*data->vectors.cols, (i + 1)*data->vectors.cols);
        for (int j = 0; j < set->T[3].rows; j++) {
            struct Slice b = Slice(set->T[3], j*set->T[3].cols, (j + 1)*set->T[3].cols);
            vector.a[j] = dot(a, b);
        }
        softmax(vector);
        double s = 0;
        const int symbol = (uint8_t)(data->labels[i]);
        for (int j = 0; j < vector.size; j++) {
            if (j == symbol) {
                s += log(vector.a[j] + .001);
            } else {
                s += log(1 - vector.a[j] + .001);
            }
        }
        sum += -s;
    }
    FreeSlice(vector);
    set->loss = sum;
    return sum;
}
double rainbow_autodiffLang(struct Set *set, struct Set *d_set, struct Data *data, struct Data *d_data) {
    return __enzyme_autodiffLang((void*)rainbowLang, set, d_set, data, d_data);
}

double Pow(double x, int i) {
    return pow(x, (double)(i+1));
}

struct Thread {
    double (*diff)(struct Set*, struct Set*, struct Data*, struct Data*);
    int offset;
    int batchSize;
    pthread_t thread;
    struct Data data;
    struct Set set;
    struct Set d_set;
    double cost;
};

void *Rainbow(void *ptr) {
    struct Thread *t = (struct Thread*)ptr;
    const int width = t->data.width;
    const int cols = t->set.cols;
    const int rows = t->set.rows;
    //printf("%d %d %d\n", width, cols, rows);
    struct Data cp = NewZeroData(width, 100);
    Within(cp.images, 99*width + width);
    Within(cp.vectors, 100*t->set.rows);
    Within(cp.entropy, 100);
    //Fprintf("offset batchsize %d %d\n", t->offset, t->batchSize);
    for (int j = t->offset; j < (t->offset + t->batchSize); j++) {
        for (int k = 0; k < 100; k++) {
            for (int l = 0; l < width; l++) {
                cp.images.a[k*width + l] = t->data.images.a[(k+j*100)*width + l];
            }
            cp.labels[k] = t->data.labels[k + j*100];
            for (int l = 0; l < rows; l++) {
                cp.vectors.a[k*rows + l] = t->data.vectors.a[(k+j*100)*rows + l];
            }
            cp.entropy.a[k] = t->data.entropy.a[k + j*100];
        }
        struct Data d_data = NewZeroData(width, 100);
        t->diff(&(t->set), &(t->d_set), &cp, &d_data);
        t->cost += t->set.loss;
        FreeData(d_data);
        for (int k = 0; k < 100; k++) {
            for (int l = 0; l < width; l++) {
                t->data.images.a[(k+j*100)*width + l] = cp.images.a[k*width + l];
            }
            t->data.labels[k + j*100] = cp.labels[k];
            for (int l = 0; l < rows; l++) {
                t->data.vectors.a[(k+j*100)*rows + l] = cp.vectors.a[k*rows + l];
            }
            t->data.entropy.a[k + j*100] = cp.entropy.a[k];
        }
    }
    FreeData(cp);
    return 0;
}

FILE *fp = NULL;
FILE *swaps = NULL;

void handler(int sig) {
    if (fp != NULL) {
        fclose(fp);
    }
    if (swaps != NULL) {
        fclose(swaps);
    }
    exit(0);
}

void mnistInference(struct Set weights) {
    struct Data data = NewData(SIZE);
    const int rows = weights.rows;
    for (int i = 0; i < 16; i++) {
        printf("calculating self entropy\n");
        struct Data cp = NewZeroData(data.width, 100);
        Within(cp.images, 99*data.width + data.width);
        Within(cp.entropy, 100);
        for (int j = 0; j < data.rows; j += 100) {
            for (int k = 0; k < 100; k++) {
                for (int l = 0; l < data.width; l++) {
                    cp.images.a[k*data.width + l] = data.images.a[(k+j)*data.width + l];
                }
                cp.labels[k] = data.labels[k + j];
                for (int l = 0; l < rows; l++) {
                    cp.vectors.a[k*rows + l] = data.vectors.a[(k+j)*rows + l];
                }
                cp.entropy.a[k] = data.entropy.a[k + j];
            }
            rainbow(&weights, &cp);
            for (int k = 0; k < 100; k++) {
                for (int l = 0; l < data.width; l++) {
                    data.images.a[(k+j)*data.width + l] = cp.images.a[k*data.width + l];
                }
                data.labels[k + j] = cp.labels[k];
                for (int l = 0; l < rows; l++) {
                    data.vectors.a[(k+j)*rows + l] = cp.vectors.a[k*rows + l];
                }
                data.entropy.a[k + j] = cp.entropy.a[k];
            }
        }
        FreeData(cp);
        if (IsSorted(data)) {
            printf("is sorted\n");
            printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[data.rows-1]);
            break;
        }
        printf("sorting\n");
        SortData(&data);
        printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[data.rows-1]);
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
}

int learn(double (*diff)(struct Set*, struct Set*, struct Data*, struct Data*), 
    struct Data data, struct Set set, int start, int epochs, int depth, double *cost) {
    const int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    const int rows = data.rows / 100;
    const int batchSize = rows / numCPU;
    const int spares = rows % numCPU;
    printf("%d %d %d\n", numCPU, batchSize, spares);
    for (int e = start; e < epochs; e++) {
        struct Set d = NewSet(set.cols, set.rows);
        *cost = 0;
        int swap = 0;
        for (int i = 0; i < depth; i++) {
            printf("calculating self entropy %d\n", i);
            Within(data.images, (data.rows - 1)*data.width + data.width);
            Within(data.vectors, data.rows*d.rows);
            Within(data.entropy, data.rows);
            struct Thread threads[numCPU];
            int j = 0;
            int cpu = 0;
            int s = 0;
            while (j < rows) {
                threads[cpu].diff = diff;
                threads[cpu].offset = j;
                threads[cpu].batchSize = batchSize;
                if (s < spares) {
                    threads[cpu].batchSize++;
                    s++;
                }
                //printf("batchSize %d\n", batchSize);
                threads[cpu].data = data;
                threads[cpu].set = set;
                threads[cpu].d_set = NewSet(set.cols, set.rows);
                threads[cpu].cost = 0;
                pthread_create(&threads[cpu].thread, 0, Rainbow, (void*)&threads[cpu]);
                j += threads[cpu].batchSize;
                cpu++;
            }
            for (int j = 0; j < numCPU; j++) {
                pthread_join(threads[j].thread, NULL);
                for (int l = 0; l < 4; l++) {
                    for (int k = 0; k < d.T[l].size; k++) {
                        d.T[l].a[k] += threads[j].d_set.T[l].a[k];
                    }
                }
                FreeSet(threads[j].d_set);
                *cost += threads[j].cost;
            }
            if (IsSorted(data)) {
                printf("is sorted\n");
                break;
            }
            printf("sorting\n");
            swap = SortData(&data);
            printf("%.17f %.17f\n", data.entropy.a[0], data.entropy.a[data.rows-1]);
        }
        double norm = 0;
        for (int s = 0; s < 4; s++) {
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
        for (int s = 0; s < 4; s++) {
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
        printf("cost %.32f, %d\n", *cost, e);
        int result = fprintf(fp, "%d %.32f\n", e, *cost);
        if (result == EOF) {
            printf("Error writing to file!\n");
            fclose(fp);
            return 1;
        }
        result = fprintf(swaps, "%d %d\n", e, swap);
        if (result == EOF) {
            printf("Error writing to swaps file!\n");
            fclose(swaps);
            return 1;
        }
    }
    printf("\n");
    return 0;
}

void load_Bible() {
    FILE *f = fopen("data/10.txt.utf-8", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    int result = fseek(f, 0, SEEK_END);
    if (result == EOF) {
        printf("Error seeking in file!\n");
        fclose(f);
        exit(1);
    }
    long fsize = ftell(f);
    BibleSize = fsize;
    result = fseek(f, 0, SEEK_SET);
    if (result == EOF) {
        printf("Error seeking in file!\n");
        fclose(f);
        exit(1);
    }
    Bible = calloc(fsize, sizeof(uint8_t));
    result = fread(Bible, fsize, 1, f);
    if (result == EOF) {
        printf("Error reading from file!\n");
        fclose(f);
        exit(1);
    }
    result = fclose(f);
    if (result == EOF) {
        printf("Error closing file!\n");
        fclose(f);
        exit(1);
    }

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            for (int k = 0; k < 256; k++) {
                Markov[i][j][k] = 0;
            }
        }
    }

    char a = 0;
    char b = 0;
    for (int i = 0; i < fsize; i++) {
        Markov[a][b][Bible[i]]++;
        a = b;
        b = Bible[i];
    }

    struct Slice buffer = MakeSlice(256);
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            double sum = 0;
            for (int k = 0; k < 256; k++) {
                double a = Markov[i][j][k];
                buffer.a[k] = a;
                sum += a*a;
            }
            //double length = sqrt(sum);
            //if (length == 0) {
            //    for (int k = 0; k < 256; k++) {
            //        Markov[i][j][k] = 1/sqrt(256);
            //    }
            //} else {
            softmax(buffer);
            for (int k = 0; k < 256; k++) {
                Markov[i][j][k] = buffer.a[k];
            }
            //}
        }
    }
    FreeSlice(buffer);
}

int main(int argc, char *argv[]) {
    srand(1);
    signal(SIGINT, handler);
    load_mnist();
    load_Bible();
    uint8_t inference = 0;
    uint8_t lang = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "-lang") == 0) {
            lang = 1;
        } else {
            inference = 1;
        }
    }
    if (inference == 1) {
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
        uint8_t *buf = calloc(fsize, sizeof(uint8_t));
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
        for (int i = 0; i < 4; i++) {
            weights.T[i].size = set->weights[i]->n_values;
            weights.T[i].cols = set->weights[i]->shape[0];
            weights.T[i].rows = set->weights[i]->shape[1];
            weights.T[i].a = set->weights[i]->values;
            weights.M[i].size = set->weights[i]->n_values;
            weights.M[i].cols = set->weights[i]->shape[0];
            weights.M[i].rows = set->weights[i]->shape[1];
            weights.M[i].a = set->weights[i]->states;
            weights.V[i].size = set->weights[i]->n_values;
            weights.V[i].cols = set->weights[i]->shape[0];
            weights.V[i].rows = set->weights[i]->shape[1];
            weights.V[i].a = set->weights[i]->states + set->weights[i]->n_values;
        }
        weights.cols = set->weights[0]->shape[0];
        weights.rows = set->weights[0]->shape[1];
        mnistInference(weights);

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
    swaps = fopen("swaps.txt", "w");
    if (swaps == NULL) {
        printf("Error opening swaps file!\n");
        return 1;
    }
    result = fprintf(swaps, "# epoch swaps\n");
    if (result == EOF) {
        printf("Error writing to swaps file!\n");
        fclose(swaps);
        return 1;
    }
    struct Set set;
    int width = 0;
    double cost = 0;
    int epochs = 0;
    if (lang == 1) {
        width = 256;
        set = NewSet(width, 32);
        for (int s = 0; s < 4; s++) {
             double factor = sqrt(2.0 / ((double)set.T[s].cols));
            for (int i = 0; i < set.T[s].size; i++) {
                set.T[s].a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
            }
        }
        const int depth = 3;
        for (epochs = 0; epochs < 8; epochs++) {
            for (int i = 0; i < 1024; i++) {
                int offset = rand() % (BibleSize - 4000);
                struct Data data = NewBibleData(offset);
                result = learn(rainbow_autodiffLang, data, set, epochs, epochs+1, depth, &cost);
                if (result > 0) {
                    return result;
                }
                FreeData(data);
            }
        }
    } else {
        struct Data data = NewData(SIZE);
        width = data.width;
        set = NewSet(data.width, 32);
        double factor = sqrt(2.0 / ((double)data.width));
        for (int s = 0; s < 4; s++) {
            for (int i = 0; i < set.T[s].size; i++) {
                set.T[s].a[i] = factor*(((double)rand() / (RAND_MAX)) * 2 - 1);
            }
        }
        epochs = 256;
        const int depth = 3;
        double cost = 0;
        result = learn(rainbow_autodiff, data, set, 0, epochs, depth, &cost);
        if (result > 0) {
            return result;
        }
        FreeData(data);
    }
    struct ProtoTf64__Set protoSet = PROTO_TF64__SET__INIT;
    struct ProtoTf64__Weights *weights[4];
    for (int i = 0; i < 4; i++) {
        struct ProtoTf64__Weights *weight = calloc(1, sizeof(struct ProtoTf64__Weights));
        *weight = (struct ProtoTf64__Weights)PROTO_TF64__WEIGHTS__INIT;
        weight->n_shape = 2;
        weight->shape = calloc(2, sizeof(int64_t));
        weight->shape[0] = width;
        weight->shape[1] = set.T[i].rows;
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
    protoSet.n_weights = 4;
    protoSet.weights = weights;
    unsigned len = proto_tf64__set__get_packed_size(&protoSet);
    void *buf = calloc(len, sizeof(uint8_t));
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
    fclose(swaps);
    FreeSet(set);
}