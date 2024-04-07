version = 16
llvm = /lib/llvm-$(version)
enzyme = ../Enzyme/enzyme/build/Enzyme/LLVMEnzyme-$(version).so

rainbow: output.ll
	$(llvm)/bin/clang $^ -lpthread -lm -O3 -o $@

output.ll: input.ll
	$(llvm)/bin/opt -load-pass-plugin=$(enzyme) -passes=enzyme -o $@ $^ -S

input.ll: main.c
	$(llvm)/bin/clang $^ -S -emit-llvm -o $@ -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

clean:
	rm -f input.ll output.ll rainbow