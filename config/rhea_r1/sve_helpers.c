#include <stdint.h>

uint64_t get_sve_byte_size()
{
    uint64_t byte_size = 0;
    __asm__ volatile(
            " mov %[byte_size],#0\n\t"
            " incb %[byte_size]\n\t"
            : [byte_size] "=r" (byte_size)
            :
            :
            ); 
    return byte_size;
}

void  adjust_sve_mr_nr_d(int* m_r, int* n_r)
{
    *m_r = (2*get_sve_byte_size())/8;
    *n_r = 8;
}

void  adjust_sve_mr_nr_s(int* m_r, int* n_r)
{
    *m_r = (2*get_sve_byte_size())/4;
    *n_r = 8;
}
