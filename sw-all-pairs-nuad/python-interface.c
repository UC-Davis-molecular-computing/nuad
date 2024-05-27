#include <sys/stat.h>
#include <assert.h>
#include "python-interface.h"


void usage() {
    fprintf(stdout, "usage: sw-score-all-pairs <match> <mismatch> <gap_open> <gap_extend> <num> <length> <sequence>\n\n");
    fprintf(stdout, "\t<match>      score for matching characters\n");
    fprintf(stdout, "\t<mismatch>   score for mismatching characters\n");
    fprintf(stdout, "\t<gap_open>   score to open a gap\n");
    fprintf(stdout, "\t<gap_extend> score to extend a gap\n");
    fprintf(stdout, "\t<num>        total number of equi-length sequences to compare\n");
    fprintf(stdout, "\t<length>     length of every sequence\n");
    fprintf(stdout, "\t<sequence>   concatenation of every sequence - must have length <num>*<length>\n");
    fprintf(stdout, "\n");
}

bool parse_cmd(int argc, const char *argv[], struct alignment_spec *spec) {
    if (argc != 8) {
        usage();
        return false;
    }

    spec->match = atoi(argv[1]);
    spec->mismatch = atoi(argv[2]);
    spec->gap_open = atoi(argv[3]);
    spec->gap_extend = atoi(argv[4]);
    spec->num = atoi(argv[5]);
    spec->length = atoi(argv[6]);
    spec->sequence = (int8_t *)argv[7];

    // assert(spec->match >= 0);
    // assert(spec->mismatch >= spec->match);
    // assert(spec->gap_open < 0);
    // assert(spec->gap_extend < 0);
    // assert(spec->length >= 1);
    // assert(spec->num >= 2);
    // assert(spec->length >= 1);
    // assert((unsigned long)(spec->num * spec->length) == strlen((const char *)spec->sequence));
        
    return true;
}


/* Returns the complement base if b is in set 'acgtACGT'.  For any
 * other character, returns 'a'. */
inline int8_t rc_base(int8_t b) {
    switch((char)b) {
    case 'a':
        return (int8_t)'t';
    case 'c':
        return (int8_t)'g';
    case 'g':
        return (int8_t)'c';
    case 't':
        return (int8_t)'a';
    case 'A':
        return (int8_t)'T';
    case 'C':
        return (int8_t)'G';
    case 'G':
        return (int8_t)'C';
    default:
        return (int8_t)'A';
    }
}

/* Writes the reverse complement of sequence beginning at src, of
 * length len, into dst.  Assumes memory for dst has been
 * pre-allocated. */
inline void rc(const int8_t *src, int len, int8_t *dst) {
    int i;
    for(i=0; i < len; ++i) {
        dst[i] = rc_base(src[i]);
    }
}

void lcs_bulk_simd(int length1, int length2, int gap_open, int gap_extend, int match, int mismatch, int num, char **seqs1, char **seqs2, int* res) {
    int i, score;

    struct alignment_spec spec;
    spec.gap_open = gap_open;
    spec.gap_extend = gap_extend;
    spec.match = match;
    spec.mismatch = mismatch;
    spec.num = num;
    // spec.sequence = (int8_t*)contents;
    int8_t *rc_seq;

    if (sw_vector_setup(length1, length2, spec.gap_open, spec.gap_extend, spec.match, spec.mismatch) != 0) {
        // fprintf(stderr, "Error initializing alignment algorithm.  Aborting.\n");
        return;
    }

    /* allocate memory to hold reverse complement sequence */
    rc_seq = malloc(length1 * sizeof(int8_t));
    for(i = 0; i < spec.num; i++) {
        // printf("%s\n%s\n", seqs1[i], seqs2[i]);
        rc((int8_t*)seqs1[i], length1, rc_seq);
        // printf("%s\n%s\n", rc_seq, seqs2[i]);
        // for(j = i + 1; j < spec.num; ++j) {
        //     boff = j * spec.length;
        //     score = sw_vector(rc_seq, 0, spec.length, spec.sequence, boff, spec.length);
        //     fprintf(ofptr, "%d %d %d\n", i, j, score);
        // }
        score = sw_vector(rc_seq, 0, length1, (int8_t*)seqs2[i], 0, length2);
        res[i] = score;
    }

    /* release allocated memory */
    sw_vector_cleanup();
    free(rc_seq);
    // free(contents);
    // fclose(ofptr);
}
