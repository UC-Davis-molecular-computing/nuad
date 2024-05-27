#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "sw-vector.h"

struct alignment_spec {
    int match;
    int mismatch;
    int gap_open;
    int gap_extend;
    int num;
    int length;
    const int8_t * sequence;
};

void usage();
bool parse_cmd(int argc, const char *argv[], struct alignment_spec *spec);
int8_t rc_base(int8_t b);
void rc(const int8_t *src, int len, int8_t *dst);
void lcs_bulk_simd(int length1, int length2, int gap_open, int gap_extend, int match, int mismatch, int num, char **seqs1, char **seqs2, int* res);
