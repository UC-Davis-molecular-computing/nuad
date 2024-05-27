#include "python-interface.h"



int main(int argc, const char *argv[]) {
    // if(!parse_cmd(argc, argv, &spec)) { return -1; };
    char** seqs1 = malloc(2 * sizeof(char *));
    char** seqs2 = malloc(2 * sizeof(char *));
    for (int i = 0; i<2; i++) {
        seqs1[i] = malloc(9*sizeof(char));
        seqs2[i] = malloc(9*sizeof(char));
    }
    strcpy(seqs1[0], "AAAAAAAA");
    strcpy(seqs1[1], "GGGGGGGG");
    strcpy(seqs2[0], "CTCTCCCT");
    strcpy(seqs2[1], "CCCTCCTC");
    int* ans = malloc(sizeof(int)*2);
    lcs_bulk_simd(8, 0, 0, 1, 0, 2, seqs1, seqs2, ans);
    printf("%d\n%d\n", ans[0], ans[1]);
    free(ans);
    for (int i = 0; i<2; i++) {
        free(seqs1[i]);
        free(seqs2[i]);
    }
    free(seqs1);
    free(seqs2);
    return 0;
}