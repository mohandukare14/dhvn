CC = gcc
CFLAGS = -O3 -march=native -mavx2 -mfma -fopenmp -fPIC -Wall
LDFLAGS = -shared -fopenmp -lm

all: lib/libdhvn.so lib/libtransformer.so lib/libdhvn_train.so lib/libdhvn_deep.so lib/libdhvn_mpm.so

lib/libdhvn.so: src/dhvn_core.c src/dhvn_core.h
	@mkdir -p lib
	$(CC) $(CFLAGS) -o $@ src/dhvn_core.c $(LDFLAGS)

lib/libtransformer.so: src/transformer_core.c src/transformer_core.h
	@mkdir -p lib
	$(CC) $(CFLAGS) -o $@ src/transformer_core.c $(LDFLAGS)

lib/libdhvn_train.so: src/dhvn_train.c src/dhvn_train.h src/dhvn_core.c src/dhvn_core.h
	@mkdir -p lib
	$(CC) $(CFLAGS) -o $@ src/dhvn_train.c src/dhvn_core.c $(LDFLAGS)

lib/libdhvn_deep.so: src/dhvn_deep.c src/dhvn_deep.h src/dhvn_core.c src/dhvn_core.h
	@mkdir -p lib
	$(CC) $(CFLAGS) -o $@ src/dhvn_deep.c src/dhvn_core.c $(LDFLAGS)

lib/libdhvn_mpm.so: src/dhvn_mpm.c src/dhvn_mpm.h src/dhvn_core.c src/dhvn_core.h
	@mkdir -p lib
	$(CC) $(CFLAGS) -o $@ src/dhvn_mpm.c src/dhvn_core.c $(LDFLAGS)

clean:
	rm -rf lib/*.so
