version = 16
llvm = /lib/llvm-$(version)
enzyme = ../Enzyme/enzyme/build/Enzyme/LLVMEnzyme-$(version).so

rainbow: output.ll
	$(llvm)/bin/clang output.ll -lpthread -lm -O3 -o rainbow

output.ll: input.ll
	$(llvm)/bin/opt -load-pass-plugin=$(enzyme) -passes=enzyme -o output.ll input.ll -S

input.ll: main.c
	$(llvm)/bin/clang main.c -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

clean:
	rm -f input.ll output.ll rainbow