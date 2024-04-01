/lib/llvm-16/bin/clang test.c -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops
/lib/llvm-16/bin/opt -load-pass-plugin=../Enzyme/enzyme/build/Enzyme/LLVMEnzyme-16.so -passes=enzyme -o output.ll input.ll -S
/lib/llvm-16/bin/clang output.ll -lm -O3 -o test