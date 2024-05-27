# CC=gcc
# CFLAGS=-c -Wall -march=native -O3 -std=c99 -DNDEBUG
# LDFLAGS=
# SOURCES=main.c sw-vector.c
# OBJECTS=$(SOURCES:.c=.o)
# EXECUTABLE=sw-score-all-pairs

# all: $(SOURCES) $(EXECUTABLE)

# $(EXECUTABLE): $(OBJECTS)
# 	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

# .c.o:
# 	$(CC) $(CFLAGS) $< -o $@

sw_score_all_pairs_test: test.o python-interface.o sw-vector.o
	gcc -g -Wall -o sw_score_all_pairs_test test.o python-interface.o sw-vector.o
lib_sw_score_all_pairs.so: python-interface.o sw-vector.o
	gcc -shared -fPIC python-interface.o sw-vector.o -o lib_sw_score_all_pairs.so
test.o: test.c python-interface.h sw-vector.h
	gcc -g -Wall -c test.c
python-interface.o: python-interface.c python-interface.h sw-vector.h
	gcc -g -fPIC -Wall -c -fPIC python-interface.c
sw-vector.o: sw-vector.c sw-vector.h
	gcc -g -fPIC -Wall -c -fPIC sw-vector.c