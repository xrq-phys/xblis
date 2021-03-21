#include "blis.h"
#include "sve_kernels.h"

void* sve_get_override_kernel_d(int kernel_idx)
{
#define UKRCASE(suffix)\
    case dukr_ ## suffix :\
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
        UKRCASE(2vx12_ld1rd)
        UKRCASE(2vx12_ld1rqd)
        UKRCASE(4vx5)
        UKRCASE(4vx5_ld1rd_colwise)
        default: return NULL;
    }
#undef UKRCASE
    return NULL;
}

#define mr_2v vec_size*2;
#define mr_4v vec_size*4;

void sve_override_mr_nr_d(int kernel_idx, int vec_size, int* m_r, int* n_r)
{
#define UKRCASEEX(mr,nr,suffix)\
    case dukr_ ## mr ## x ## nr ## suffix :\ 
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
        UKRCASEEX(2v,12,_ld1rd)
        UKRCASEEX(2v,12,_ld1rqd)
        UKRCASEEX(4v,5,)
        UKRCASEEX(4v,5,_ld1rd_colwise)
    }
}

void* sve_get_override_kernel_z(int kernel_idx)
{
#define UKRCASE(suffix)\
    case zukr_ ## suffix :\
        {\
            return bli_zgemm_armv8a_sve_asm_ ## suffix ;\
        }
    switch(kernel_idx)
    {
        UKRCASE(2vx4)
        UKRCASE(2vx5)
        UKRCASE(2vx6)
        UKRCASE(2vx8)
        UKRCASE(2vx10)
        UKRCASE(2vx12)
        default: return NULL;
    }
#undef UKRCASE
    return NULL;
}

void sve_override_mr_nr_z(int kernel_idx, int vec_size, int* m_r, int* n_r)
{
#define UKRCASEEX(mr,nr,suffix)\
    case zukr_ ## mr ## x ## nr ## suffix :\ 
        {\
            *m_r = mr_ ## mr;\
            *n_r = nr;\
            return;\
        }
    switch(kernel_idx)
    {
        UKRCASEEX(2v,4,)
        UKRCASEEX(2v,5,)
        UKRCASEEX(2v,6,)
        UKRCASEEX(2v,8,)
        UKRCASEEX(2v,10,)
        UKRCASEEX(2v,12,)
    }
}

void* sve_get_override_kernel_c(int kernel_idx)
{
#define UKRCASE(suffix)\
    case zukr_ ## suffix :\
        {\
            return bli_cgemm_armv8a_sve_asm_ ## suffix ;\
        }
    switch(kernel_idx)
    {
        UKRCASE(2vx10)
        default: return NULL;
    }
#undef UKRCASE
    return NULL;
}

void sve_override_mr_nr_c(int kernel_idx, int vec_size, int* m_r, int* n_r)
{
#define UKRCASEEX(mr,nr,suffix)\
    case cukr_ ## mr ## x ## nr ## suffix :\ 
        {\
            *m_r = mr_ ## mr;\
            *n_r = nr;\
            return;\
        }
    switch(kernel_idx)
    {
        UKRCASEEX(2v,10,)
    }
}
