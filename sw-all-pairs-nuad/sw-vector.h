/* 
   Original code from SHRiMP aligner (http://compbio.cs.toronto.edu/shrimp/).
   Modified by Chris Thachuk, 2015.
*/
/*	$Id: sw-vector.h,v 1.7 2009/06/16 23:26:21 rumble Exp $	*/
#include <stdbool.h>
#include <stdint.h>

#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))

int sw_vector_cleanup(void);
int sw_vector_setup(int, int, int, int, int, int);
int sw_vector(const int8_t *, int, int, const int8_t *, int, int);
