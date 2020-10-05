#include "blis.h"
#include "sve_kernels.h"

void* sve_get_override_kernel_d(int kernel_idx)
{
#define UKRCASE(suffix)\
    case ukr_ ## suffix :\
        {\
            return bli_dgemm_armv8a_sve_asm_ ## suffix ;\
        }
    switch(kernel_idx)
    {
        UKRCASE(2vx8)
        UKRCASE(2vx8_ld1rd)
        UKRCASE(2vx8_ld1rqd)
        UKRCASE(2vx9)
        UKRCASE(2vx10_ld1rd)
        UKRCASE(2vx10_ld1rd_colwise)
        UKRCASE(2vx10_ld1rqd)
        UKRCASE(2vx12)
        UKRCASE(2vx12_dup)
        UKRCASE(2vx12_ld1rqd)
        UKRCASE(4vx5)
        UKRCASE(4vx5_ld1rd_colwise)
        default: return NULL;
    }
#undef UKRCASE
    return NULL;
}

void sve_override_mr_nr_d(int kernel_idx, int vec_size, int* m_r, int* n_r)
{
#define mr_2v vec_size*2;
#define mr_4v vec_size*4;
#define UKRCASEEX(mr,nr,suffix)\
    case ukr_ ## mr ## x ## nr ## suffix :\ 
        {\
            *m_r = mr_ ## mr;\
            *n_r = nr;\
            return;\
        }
    switch(kernel_idx)
    {
        UKRCASEEX(2v,8,)
        UKRCASEEX(2v,8,_ld1rd)
        UKRCASEEX(2v,8,_ld1rqd)
        UKRCASEEX(2v,9,)
        UKRCASEEX(2v,10,_ld1rd)
        UKRCASEEX(2v,10,_ld1rd_colwise)
        UKRCASEEX(2v,10,_ld1rqd)
        UKRCASEEX(2v,12,)
        UKRCASEEX(2v,12,_dup)
        UKRCASEEX(2v,12,_ld1rqd)
        UKRCASEEX(4v,5,)
        UKRCASEEX(4v,5,_ld1rd_colwise)
    }
}
