def reverse_complement(dna):
    complement_dict = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}
    reverse_comp = ''.join(complement_dict[base] for base in dna)
    return reverse_comp

def longest_common_subsequence(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0 if i == 0 or j == 0 else None for j in range(n+1)] for i in range(m+1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    return lcs_length
