#ifndef BLIS_SVE_HELPERS_H
#define BLIS_SVE_HELPERS_H

#include <stdint.h>

uint64_t get_sve_byte_size();

void adjust_sve_mr_nr_d(int* m_r, int* n_r);
void adjust_sve_mr_nr_s(int* m_r, int* n_r);

#endif // BLIS_SVE_HELPERS_H
