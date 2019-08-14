#if defined(DEBUG)
#include "bli_gemm_sve_asm_macros.h"
#include <stdio.h>

void print_marker(uint64_t val)
{
    printf("Marker %lu\n",val);
}

void print_pointer(void* p)
{
    printf("Pointer: 0x%p\n", p);
}

void print_counter(uint64_t counter, uint64_t inc)
{
    static int initialized = 0;
    static uint64_t counters[MAX_COUNTERS];
    if(!initialized)
    {
        for(int i = 0; i < MAX_COUNTERS; i++)
        {
            counters[i] = 0;
        }
        initialized = 1;
    }
    printf("Counter %lu: %lu\n", counter, counters[counter]);
    counters[counter] += inc;
}

void print_dvector(double* ptr, uint64_t nelem)
{
    printf("Vector content: {");
    for(uint64_t i = 0; i < nelem-1; i++)
    {
        printf("%.5e, ",ptr[i]);
    }
    printf("%.5e}\n",ptr[nelem-1]);
}

void print_slvector(int64_t* ptr, uint64_t nelem)
{
    printf("Vector content: {");
    for(uint64_t i = 0; i < nelem-1; i++)
    {
        printf("%ld, ",ptr[i]);
    }
    printf("%ld}\n",ptr[nelem-1]);
}
#endif
