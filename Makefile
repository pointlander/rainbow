version = 16
llvm = /lib/llvm-$(version)
enzyme = ../Enzyme/enzyme/build/Enzyme/LLVMEnzyme-$(version).so

rainbow: output.ll set.ll
	$(llvm)/bin/clang $^ -lprotobuf-c -lpthread -lm -O3 -o $@

output.ll: input.ll
	$(llvm)/bin/opt -load-pass-plugin=$(enzyme) -passes=enzyme -o $@ $^ -S

input.ll: main.c set.pb-c.h mnist/mnist.h
	$(llvm)/bin/clang $< -S -emit-llvm -o $@ -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

set.ll: set.pb-c.c set.pb-c.h
	$(llvm)/bin/clang $< -S -emit-llvm -o $@ -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops

set.pb-c.c set.pb-c.h: set.proto
	protoc --c_out=. $<

clean:
	rm -f input.ll output.ll rainbow