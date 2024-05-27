/* 
   Original code from SHRiMP aligner (http://compbio.cs.toronto.edu/shrimp/).
   Modified by Chris Thachuk, 2015.
*/

#include "sw-vector.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>

#include <mmintrin.h>	/* MMX */
#include <xmmintrin.h>	/* SSE */
#include <emmintrin.h>	/* SSE2 */
#include <smmintrin.h>
#include <sys/time.h>

static int	initialised;
static int8_t  *db, *qr;
static int	dblen, qrlen;
static int16_t *nogap, *b_gap;
static int	a_gap_open, a_gap_ext;
static int	b_gap_open, b_gap_ext;
static int	match, mismatch;

/* static inline int16_t hmax(__m128i buffer) { */
/*     __m128i tmp1 = _mm_sub_epi8(_mm_set1_epi8(127), buffer); */
/*     __m128i tmp2 = _mm_min_epu8(tmp1, _mm_srli_epi16(tmp1, 8)); */
/*     __m128i tmp3 = _mm_minpos_epu16(tmp2); */
/*     return (int8_t)(127 - _mm_cvtsi128_si32(tmp3)); */
/* } */

/* static inline int16_t hMin(__m128i buffer) { */
/*     buffer = _mm_min_epi8(buffer, _mm_shuffle_epi8(buffer, m1)); */
/*     buffer = _mm_min_epi8(buffer, _mm_shuffle_epi8(buffer, m2)); */
/*     buffer = _mm_min_epi8(buffer, _mm_shuffle_epi8(buffer, m3)); */
/*     buffer = _mm_min_epi8(buffer, _mm_shuffle_epi8(buffer, m4)); */
/*     return ((int8_t*) ((void *) &buffer))[0]; */
/* } */


/*
 * Calculate the Smith-Waterman score.
 *
 * This is basically an SSE2 version of Wozniak's vectored implementation, but
 * without a score table. Further, we assume a fixed database and query size,
 * so *nogap and *b_gap must be pre-allocated (the malloc overhead for very
 * small scans is _huge_).
 *
 * NOTE THE FOLLOWING:
 *
 *	1) seqA must be padded with 7 bytes at the beginning and end. The first
 *	   element of seqA should be the first pad byte.
 *
 *	2) seqB must be padded with bytes on the end up to mod 8 characters.
 *	   The first element of seqB should be (of course) the first character.
 *
 *	3) seqA and seqB's padding _must_ be different, otherwise our logic will
 *	   consider the padding as matches!
 *
 *      4) These is no _mm_max_epu16 prior to SSE 4! We must use the signed max
 *         function. Unfortunately, this limits our maximum score to 2^15 - 1, or
 *         32767. Since bad things happen if we roll over, our caller must ensure
 *         that this will not happen.
 */

static inline int vect_sw_same_gap(int8_t *seqA, int lena, int8_t *seqB, int lenb) {
    int i, j, score = 0;
    __m128i v_score, v_zero, v_match, v_mismatch;
    __m128i v_a_gap_ext, v_a_gap_open_ext;
    __m128i v_a_gap, v_b_gap, v_nogap;
    __m128i v_last_nogap, v_prev_nogap, v_seq_a, v_seq_b;
    __m128i v_tmp;

#define SET16(a, e7, e6, e5, e4, e3, e2, e1, e0)        \
    _mm_set_epi16((int16_t)a[e7], (int16_t)a[e6],       \
                  (int16_t)a[e5], (int16_t)a[e4],       \
                  (int16_t)a[e3], (int16_t)a[e2],       \
                  (int16_t)a[e1], (int16_t)a[e0])

    v_score		 = _mm_setzero_si128();
    v_zero		 = _mm_setzero_si128();
    v_match		 = SET16((&match), 0, 0, 0, 0, 0, 0, 0, 0);
    v_mismatch	 = SET16((&mismatch), 0, 0, 0, 0, 0, 0, 0, 0);
    v_a_gap_ext	 = SET16((&a_gap_ext), 0, 0, 0, 0, 0, 0, 0, 0);
    v_a_gap_open_ext = SET16((&a_gap_open), 0, 0, 0, 0, 0, 0, 0, 0);
    v_a_gap_open_ext = _mm_add_epi16(v_a_gap_open_ext, v_a_gap_ext);

    for (i = 0; i < lena + 14; i++) {
        nogap[i] = 0;
        b_gap[i] = (int16_t)-b_gap_open;
    }

    for (i = 0; i < (lenb + 7)/8; i++) {
        int k = i * 8;

        v_b_gap = SET16(b_gap, 6, 6, 5, 4, 3, 2, 1, 0);
        v_nogap = SET16(nogap, 6, 6, 5, 4, 3, 2, 1, 0);
        v_seq_a = SET16(seqA, 0, 0, 1, 2, 3, 4, 5, 6);
        v_seq_b = SET16(seqB, k+7, k+6, k+5, k+4, k+3, k+2, k+1, k+0);

        v_a_gap = v_a_gap_ext;
        v_a_gap = _mm_sub_epi16(v_a_gap, v_a_gap_open_ext);

        v_last_nogap = _mm_setzero_si128();
        v_prev_nogap = _mm_setzero_si128();

        for (j = 0; j < (lena + 7); j++) {
            v_b_gap = _mm_slli_si128(v_b_gap, 2);
            v_b_gap = _mm_insert_epi16(v_b_gap, b_gap[j+7], 0);

            v_nogap = _mm_slli_si128(v_nogap, 2);
            v_nogap = _mm_insert_epi16(v_nogap, nogap[j+7], 0);

            v_seq_a = _mm_slli_si128(v_seq_a, 2);
            v_seq_a = _mm_insert_epi16(v_seq_a, seqA[j+7], 0);

            v_tmp = _mm_sub_epi16(v_last_nogap, v_a_gap_open_ext);
            v_a_gap = _mm_sub_epi16(v_a_gap, v_a_gap_ext);
            v_a_gap = _mm_max_epi16(v_a_gap, v_tmp);

            v_tmp = _mm_sub_epi16(v_nogap, v_a_gap_open_ext);
            v_b_gap = _mm_sub_epi16(v_b_gap, v_a_gap_ext);
            v_b_gap = _mm_max_epi16(v_b_gap, v_tmp);

            /* compute the score (v_last_nogap is a tmp variable) */
            v_last_nogap = _mm_cmpeq_epi16(v_seq_a, v_seq_b);
            v_tmp = _mm_and_si128(v_last_nogap, v_match);
            v_last_nogap = _mm_cmpeq_epi16(v_last_nogap, v_zero);
            v_last_nogap = _mm_and_si128(v_last_nogap, v_mismatch);
            v_tmp = _mm_or_si128(v_tmp, v_last_nogap);

            v_last_nogap = _mm_add_epi16(v_prev_nogap, v_tmp);
            v_last_nogap = _mm_max_epi16(v_last_nogap, v_zero);
            v_last_nogap = _mm_max_epi16(v_last_nogap, v_a_gap);
            v_last_nogap = _mm_max_epi16(v_last_nogap, v_b_gap);
			
            v_prev_nogap = v_nogap;
            v_nogap = v_last_nogap;

            b_gap[j] = (int16_t)_mm_extract_epi16(v_b_gap, 7);
            nogap[j] = (int16_t)_mm_extract_epi16(v_nogap, 7);

            v_score = _mm_max_epi16(v_score, v_last_nogap);
        }
    }

    // assert(score == 0);
    /* score = hmax(v_score); */
    score = MAX(score, _mm_extract_epi16(v_score, 0));
    score = MAX(score, _mm_extract_epi16(v_score, 1));
    score = MAX(score, _mm_extract_epi16(v_score, 2));
    score = MAX(score, _mm_extract_epi16(v_score, 3));
    score = MAX(score, _mm_extract_epi16(v_score, 4));
    score = MAX(score, _mm_extract_epi16(v_score, 5));
    score = MAX(score, _mm_extract_epi16(v_score, 6));
    score = MAX(score, _mm_extract_epi16(v_score, 7));

    return (score);
}

int sw_vector_cleanup(void) {
	free(db);
	free(qr);
	free(nogap);
	free(b_gap);
	return 0;
}

int sw_vector_setup(int _dblen, int _qrlen, int _a_gap_open, int _a_gap_ext, int _match, int _mismatch) {
    if (_match * _qrlen >= 32768) {
        fprintf(stderr, "Error: Match Value is too high/reads are too long. "
                "Please ensure that (Match_Value x your_longest_read_length)"
                " is less than 32768! Try using smaller S-W values.");
        exit(1);
    }

    dblen = _dblen;
    db = (int8_t *)malloc((dblen + 14) * sizeof(db[0]));
    if (db == NULL)
        return (1);

    qrlen = _qrlen;
    qr = (int8_t *)malloc((qrlen + 14) * sizeof(qr[0]));
    if (qr == NULL)
        return (1);

    nogap = (int16_t *)malloc((dblen + 14) * sizeof(nogap[0]));
    if (nogap == NULL)
        return (1);

    b_gap = (int16_t *)malloc((dblen + 14) * sizeof(b_gap[0]));
    if (b_gap == NULL)
        return (1);

    a_gap_open = -(_a_gap_open);
    a_gap_ext  = -(_a_gap_ext);
    b_gap_open = -(_a_gap_open);
    b_gap_ext  = -(_a_gap_ext);
    match = _match;
    mismatch = _mismatch;

    initialised = 1;

    return (0);
}

inline int sw_vector(const int8_t *seqa, int aoff, int alen, const int8_t *seqb, int boff, int blen) {
    int score;

    // assert(initialised);
    // assert(alen > 0 && alen <= dblen);
    // assert(blen > 0 && blen <= qrlen);

    memset(db, -1, (dblen + 14) * sizeof(db[0]));
    memset(qr, -2, (qrlen + 14) * sizeof(qr[0]));

    memcpy(&db[7], &seqa[aoff], alen);
    memcpy(&qr[7], &seqb[boff], blen);

    score = vect_sw_same_gap(&db[0], alen, &qr[7], blen);

    return (score);
}
