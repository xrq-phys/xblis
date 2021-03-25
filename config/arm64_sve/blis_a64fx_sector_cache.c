#include "bli_a64fx_sector_cache.h"

void a64fx_print_sc_configuration(uint64_t bitfield)
{
    for(uint64_t sector = 0; sector < 4; sector++)
    {
        uint64_t count = bitfield & 0x7;
        bitfield >>= 4;
        printf("cache sector %zu has %zu sectors\n", sector, count+1LU);
    }
}
