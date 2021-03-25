#ifndef BLIS_A64FX_SECTOR_CACHE_H
#define BLIS_A64FX_SECTOR_CACHE_H

#if defined(_A64FX)
    // A64FX: set up cache sizes
    //
    // Reference: A64FX (TM) specification Fujitsu HPC Extension
    // Link:      https://github.com/fujitsu/A64FX/blob/master/doc/A64FX_Specification_HPC_Extension_v1_EN.pdf
    //
    // 63:15 |    14:12    |  11  |    10:08    |  07  |    06:04    |  03  |    02:00    |
    // RES0  | l1_sec3_max | RES0 | l1_sec2_max | RES0 | l1_sec1_max | RES0 | l1_sec0_max |
    //
    // the bits set number of maximum sectors from 0-7
    // 000 - 0
    // 001 - 1
    // 010 - 2
    // 011 - 3
    // 100 - 4
    // 101 - 5
    // 110 - 6
    // 111 - 7
    //
    // For L1 we want to maximize the number of sectors for B
    // Configuration 1: 1 sector for  C (sector 3)
    //                  1 sector for  A (sector 1)
    //                  6 sectors for B (sector 2)
    //                  0 sectors for the rest (sector 0)
    // 
    // 16b bitfield conf. 1: 0b0 001 0 110 0 001 0 000
    //
    // Configuration 2: 1 sector for  C (sector 3)
    //                  1 sector for  A (sector 1)
    //                  5 sectors for B (sector 2)
    //                  1 sectors for the rest (sector 0)
    // 
    // 16b bitfield conf. 2: 0b0 001 0 101 0 001 0 001
    //
    // accessing the control register:
    //
    // MRS <Xt>, S3_3_C11_C8_2
    // MSR S3_3_C11_C8_2, <Xt>
    //
    // TODO: First tests showed no change in performance, a deeper investigation
    //       is necessary
#define A64FX_SETUP_SECTOR_CACHE_SIZES(config_bitfield)\
{\
    uint64_t sector_cache_config = config_bitfield;\
    __asm__ volatile(\
            "msr s3_3_c11_c8_2,%[sector_cache_config]"\
            :\
            : [sector_cache_config] "r" (sector_cache_config)\
            :\
            );\
}

#define A64FX_SETUP_SECTOR_CACHE_SIZES_L2(config_bitfield)\
{\
    uint64_t sector_cache_config = config_bitfield;\
    __asm__ volatile(\
            "msr s3_3_c15_c8_2,%[sector_cache_config]"\
            :\
            : [sector_cache_config] "r" (sector_cache_config)\
            :\
            );\
}


#define A64FX_SET_CACHE_SECTOR(areg, tag, sparereg)\
" mov "#sparereg", "#tag"      \n\t"\
" lsl "#sparereg", "#sparereg", 56  \n\t"\
" orr "#areg", "#areg", "#sparereg"   \n\t"

#define A64FX_READ_SECTOR_CACHE_SIZES(output_uint64)\
__asm__ volatile(\
        "mrs %["#output_uint64"],s3_3_c11_c8_2"\
        : [output_uint64] "=r" (output_uint64)\
        : \
        :\
        );

#define A64FX_SCC(sec0,sec1,sec2,sec3)\
    (uint64_t)((sec0 & 0x7LU) | ((sec1 & 0x7LU) << 4) | ((sec2 & 0x7LU) << 8) | ((sec3 & 0x7LU) << 12))

#define A64FX_SCC_L2(sec02,sec13)\
    (uint64_t)((sec02 & 0x1FLU) | ((sec13 & 0x1FLU) << 8))
#else
#define A64FX_SETUP_SECTOR_CACHE_SIZES(config_bitfield) 
#define A64FX_SETUP_SECTOR_CACHE_SIZES_L2(config_bitfield) 
#define A64FX_READ_SECTOR_CACHE_SIZES(output_uint64) 
#define A64FX_SET_CACHE_SECTOR(areg,tag,sparereg) 
#define A64FX_SCC(sec0,sec1,sec2,sec3)
#define A64FX_SCC_L2(sec0,sec1,sec2,sec3)
#endif

void a64fx_print_sc_configuration(uint64_t bitfield);

#endif // BLIS_A64FX_SECTOR_CACHE_H
