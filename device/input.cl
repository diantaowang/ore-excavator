#include "param.h"

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__constant ulong blake_iv[] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

#define mix(va, vb, vc, vd, x, y) \
va = (va + vb + x); \
vd = rotate((vd ^ va), (ulong)64 - 32); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 24); \
va = (va + vb + y); \
vd = rotate((vd ^ va), (ulong)64 - 16); \
vc = (vc + vd); \
vb = rotate((vb ^ vc), (ulong)64 - 63);

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void kernel_round0(__global ulong * restrict blake_state, 
        __global ulong4 * restrict ht,   
        __global uint   * restrict rowCounters, 
        __global uint   * restrict debug)
{
    uint            input_end = 1 << 21;
    uint            dropped = 0;
    ulong           blake_state_reg[8];
    ulong           v[16];
    __local  uint   rowCsSrc[1 << 17];   

    for (uint i = 0; i < (1 << 17); ++i)
        rowCsSrc[i] = 0;

    #pragma unroll
    for (uint i = 0; i < 8; ++i)
        blake_state_reg[i] = blake_state[i];

    #pragma unroll 1
    #pragma ivdep array(ht)
    #pragma ivdep array(rowCsSrc)
    for (uint input = 0; input < input_end; ++input) {
        // shift "i" to occupy the high 32 bits of the second ulong word in the
        // message block
        ulong word1 = (ulong) (input >> 1) << 32;
        // init vector v
        v[0] = blake_state_reg[0];
        v[1] = blake_state_reg[1];
        v[2] = blake_state_reg[2];
        v[3] = blake_state_reg[3];
        v[4] = blake_state_reg[4];
        v[5] = blake_state_reg[5];
        v[6] = blake_state_reg[6];
        v[7] = blake_state_reg[7];
        v[8] = blake_iv[0];
        v[9] = blake_iv[1];
        v[10] = blake_iv[2];
        v[11] = blake_iv[3];
        v[12] = blake_iv[4];
        v[13] = blake_iv[5];
        v[14] = blake_iv[6];
        v[15] = blake_iv[7];
        // mix in length of data
        v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */ ;
        // last block
        v[14] ^= (ulong) - 1;

        // round 1
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 2
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 3
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, word1);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 4
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, word1);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 5
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, word1);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 6
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], word1, 0);
        // round 7
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], word1, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 8
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, word1);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 9
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], word1, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 10
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], word1, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 11
        mix(v[0], v[4], v[8],  v[12], 0, word1);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], 0, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);
        // round 12
        mix(v[0], v[4], v[8],  v[12], 0, 0);
        mix(v[1], v[5], v[9],  v[13], 0, 0);
        mix(v[2], v[6], v[10], v[14], 0, 0);
        mix(v[3], v[7], v[11], v[15], 0, 0);
        mix(v[0], v[5], v[10], v[15], word1, 0);
        mix(v[1], v[6], v[11], v[12], 0, 0);
        mix(v[2], v[7], v[8],  v[13], 0, 0);
        mix(v[3], v[4], v[9],  v[14], 0, 0);

        // compress v into the blake state; this produces the 50-byte hash
        // (two Xi values)
        ulong h[7];
        h[0] = blake_state_reg[0] ^ v[0] ^ v[8];
        h[1] = blake_state_reg[1] ^ v[1] ^ v[9];
        h[2] = blake_state_reg[2] ^ v[2] ^ v[10];
        h[3] = blake_state_reg[3] ^ v[3] ^ v[11];
        h[4] = blake_state_reg[4] ^ v[4] ^ v[12];
        h[5] = blake_state_reg[5] ^ v[5] ^ v[13];
        h[6] = (blake_state_reg[6] ^ v[6] ^ v[14]) & 0xffff;

        // store the two Xi values in the hash table
        uint            row;
        ulong           xi[4];

        if (input & 0x1) {
            row = ((h[3] >> 8) & 0xffff) | (((h[3] >> 8) & 0xf00000) >> 4);
            xi[0] = (ulong) input;
            xi[1] = (h[3] >> 24) | (h[4] << (64 - 24));
            xi[2] = (h[4] >> 24) | (h[5] << (64 - 24));
            xi[3] = (h[5] >> 24) | (h[6] << (64 - 24));
        } else {
            row = (h[0] & 0xffff) | ((h[0] & 0xf00000) >> 4);
            xi[0] = (ulong) input;
            xi[1] = (h[0] >> 16) | (h[1] << (64 - 16));
            xi[2] = (h[1] >> 16) | (h[2] << (64 - 16));
            xi[3] = (h[2] >> 16) | (h[3] << (64 - 16));
        }

        uint rowIdx = row >> 3;
        uint rowOffset = BITS_PER_ROW * (row & 0x7);
        uint xcnt = atom_add(rowCsSrc + rowIdx, 1 << rowOffset);
        uint cnt = (xcnt >> rowOffset) & ROW_MASK;
        if (cnt >= NR_SLOTS)
            atom_sub(rowCsSrc + rowIdx, 1 << rowOffset);
        else {
            uint htOffset = row * NR_SLOTS + cnt;
            ht[htOffset] = (ulong4)(xi[0], xi[1], xi[2],xi[3]);
        }
    }
    #pragma unroll 16
    for (uint i = 0; i < (1 << 17); ++i)
        rowCounters[i] = rowCsSrc[i];
#ifdef ENABLE_DEBUG
    debug[tid * 2] = 0;
    debug[tid * 2 + 1] = dropped;
#endif
}

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 12) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f))
#define DECODE_ROW(REF)   (REF >> 12)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#define RSHIFT8(dst, src) \
    dst.s0 = (src.s0 >> 8) | (src.s1 << (64 - 8)); \
dst.s1 = (src.s1 >> 8) | (src.s2 << (64 - 8)); \
dst.s2 = (src.s2 >> 8) | (src.s3 << (64 - 8)); \
dst.s3 = (src.s3 >> 8); \


__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel_round(__global uint * restrict roundNum,
        __global ulong4 * restrict ht_src, 
        __global ulong4 * restrict ht_dst,
        __global uint   * restrict debug,
        __global uint   * restrict rowCountersSrc, 
        __global uint   * restrict rowCountersDst)
{
    // uint         dropped_coll = 0;
    // uint         dropped_stor = 0;
    uint            round = *roundNum;
    uint            cntSrc;
    
    __local  uint   rowCsSrc[1 << 17]; 
    __local  uint   rowCsDst[1 << 17];

    ulong4          rowHash[NR_SLOTS];
    ulong4          collision;

    bool load = 1;
    uint load_num = 0;
    uint load_i, load_j;
    uint coll_num;

    #pragma unroll 16
    for (uint i = 0; i < (1 << 17); ++i)
        rowCsSrc[i] = rowCountersSrc[i];

    for (uint i = 0; i < (1 << 17); ++i)
        rowCsDst[i] = 0;

    #pragma ivdep
    for (uint tid = 0; tid < (1 << 20); ) {
        if (load) {
            uint rowIdx = tid >> 3;
            uint rowOffset = BITS_PER_ROW * (tid & 0x7);
            uint rowCounter = rowCsSrc[rowIdx];
            cntSrc = (rowCounter >> rowOffset) & ROW_MASK;
            if (cntSrc > NR_SLOTS)
                cntSrc = NR_SLOTS;
            load_num = 0;
            load_i = 0;
            load_j = 1;
            coll_num = 0;
            load = 0;
        }

        if (cntSrc > 1) {
            if (load_num < cntSrc) {
                uint htOffset = tid * NR_SLOTS + load_num;
                rowHash[load_num++] = ht_src[htOffset];
            }

            bool store;
           
            // Is "coll_num < COLL_DATA_SIZE_PER_TH" must ?
            if (load_num > 1 && load_i < cntSrc - 1 && coll_num < COLL_DATA_SIZE_PER_TH) {
                uint idx;
                uint newIndex = ENCODE_INPUTS(tid, load_i, load_j);
                ulong4 x = rowHash[load_i];
                ulong4 y = rowHash[load_j];
                ulong4 collResult = x ^ y;
                
                if (!(round & 0x1)) {
                    collResult.s1 = (collResult.s1 >> 8) | (collResult.s2 << (64 - 8));
                    collResult.s2 = (collResult.s2 >> 8) | (collResult.s3 << (64 - 8));
                    collResult.s3 = (collResult.s3 >> 8);
                }

                store = 0;
                if (round == 1 || round == 2) {
                    idx = collResult.s1 & 0xffffffff;
                    if (collResult.s1 || collResult.s2 || collResult.s3)
                        store = 1;
                } else if (round == 3 || round == 4) {
                    idx = (collResult.s1 >> 32);
                    if ((collResult.s1 & 0xffffffff00000000) || collResult.s2 || collResult.s3)
                        store = 1;
                } else if (round == 5 || round == 6) {
                    idx = collResult.s2 & 0xffffffff;
                    if (collResult.s2 || collResult.s3)
                        store = 1;
                } else if (round == 7 || round == 8) {
                    idx = (collResult.s2 >> 32);
                    if ((collResult.s2 & 0xffffffff00000000) || collResult.s3)
                        store = 1;
                }
                
                if (store) {
                    if (round & 0x1) {
                        collResult.s1 = (collResult.s1 >> 16) | (collResult.s2 << (64 - 16));
                        collResult.s2 = (collResult.s2 >> 16) | (collResult.s3 << (64 - 16));
                        collResult.s3 = (collResult.s3 >> 16);
                    } else {
                        collResult.s3 = (collResult.s3 << 16) | (collResult.s2 >> (64 - 16));
                        collResult.s2 = (collResult.s2 << 16) | (collResult.s1 >> (64 - 16));
                        collResult.s1 = (collResult.s1 << 16);
                    }
                    
                    if (round == 1) {
                        uint i = x.s0 & 0xffffffff;
                        uint j = y.s0 & 0xffffffff;
                        collision = collResult;
                        collision.s0 = (ulong) i << 32 | j;
                    }           

                    uint row; 
                    if (round & 0x1)
                        row = ((idx & 0xf0000) >> 0) |
                            ((idx & 0xf00) << 4) | ((idx & 0xf00000) >> 12) |
                            ((idx & 0xf) << 4) | ((idx & 0xf000) >> 12);
                    else 
                        row = (idx & 0xffff) | ((idx & 0xf00000) >> 4);
                    
                    uint rowIdx = row >> 3;
                    uint rowOffset = BITS_PER_ROW * (row & 0x7);
                    uint xcnt = atom_add(rowCsDst + rowIdx, 1 << rowOffset);
                    uint cnt = (xcnt >> rowOffset) & ROW_MASK;
                    ulong4 oldColl = 0;
                    uint htOffset = row * NR_SLOTS + cnt;
                    if (round != 1 && round != 2)
                        oldColl = ht_dst[htOffset];
                    if (round == 2 || round == 3) {
                        collision.s0 = oldColl.s0;
                        collision.s1 = (collResult.s1 & 0xffffffff00000000) | newIndex;
                        collision.s2 = collResult.s2;
                        collision.s3 = collResult.s3;
                    } else if (round == 4 || round == 5) {
                        collision.s0 = oldColl.s0;
                        collision.s1 = ((ulong) newIndex << 32) | (oldColl.s1 & 0xffffffff);
                        collision.s2 = collResult.s2;
                        collision.s3 = collResult.s3;
                    
                    } else if (round == 6 || round == 7) {
                        collision.s0 = oldColl.s0;
                        collision.s1 = oldColl.s1;
                        collision.s2 = (collResult.s2 & 0xffffffff00000000) | newIndex;
                        collision.s3 = collResult.s3;
                    
                    } else if (round == 8) {
                        collision.s0 = oldColl.s0;
                        collision.s1 = oldColl.s1;
                        // collision.s2 has 4 bytes garbage data in high position.
                        collision.s2 = oldColl.s2;
                        collision.s3 = (collResult.s3 << 32) | newIndex;
                    }

                    if (cnt >= NR_SLOTS)
                        atom_sub(rowCsDst + rowIdx, 1 << rowOffset);
                    else
                        ht_dst[htOffset] = collision;
                }

                if (load_j == cntSrc - 1) {
                    ++load_i;
                    load_j = load_i + 1;
                } else {
                    ++load_j;
                }
                ++coll_num;
            }
            if (load_i == cntSrc - 1 || coll_num == COLL_DATA_SIZE_PER_TH) {
                load = 1;
                ++tid;
            }  
        } else {
            load = 1;
            ++tid;
        }
    }

    #pragma unroll 16
    for (uint i = 0; i < (1 << 17); ++i)
        rowCountersDst[i] = rowCsDst[i];
}

void potential_sol(__global ulong ** restrict htabs, __global sols_t * restrict sols,
        uint ref0, uint ref1, uint * sols_nr)
{
    uint    values[1 << PARAM_K];
    uint    values_tmp[1 << (PARAM_K - 1)];
    uint    nr_value;
    bool    dup_value = 0;
    int     dup_to_watch = -1;

    values_tmp[0] = ref0;
    values_tmp[1] = ref1;
    nr_value = 2;
    
    // k -> the round of load global memory. 
    for (uint k = PARAM_K - 2; k > 1; --k) {
        __global ulong *ht = htabs[k & 0x1];
        uint i = nr_value - 1;
        uint j = nr_value * 2 - 1;
        uint offset;
        bool lowHalf;
        if (k ==2 || k == 3) {
            offset = 1;
            lowHalf = 1;
        } else if (k == 4 || k == 5) {
            offset = 1;
            lowHalf = 0;
        } else if (k == 6 || k == 7) {
            offset = 2;
            lowHalf = 1;
        }

        #pragma ivdep array(values_tmp)
        do {
            uint ins_pre = values_tmp[i];
            uint row = DECODE_ROW(ins_pre);
            uint solt0 = DECODE_SLOT0(ins_pre);
            uint solt1 = DECODE_SLOT1(ins_pre);

            uint rowBase = row * NR_SLOTS * 4;

            ulong t0 = ht[rowBase + solt0 * 4 + offset];
            ulong t1 = ht[rowBase + solt1 * 4 + offset];

            uint x, y;
            if (lowHalf) {
                x = t0 & 0xffffffff;
                y = t1 & 0xffffffff; 
            } else {
                x = t0 >> 32;
                y = t1 >> 32;
            }

            values_tmp[j] = y;
            values_tmp[j - 1] = x;

            if (!i)
                break;
            --i;
            j -= 2;
        } while(1);
        nr_value *= 2;
    }
    
    __global ulong *ht = htabs[1];
    uint i = nr_value - 1;
    uint j = nr_value * 4 - 1;
    #pragma ivdep array(values)
    do {
        uint ins_pre = values_tmp[i];
        uint row = DECODE_ROW(ins_pre);
        uint solt0 = DECODE_SLOT0(ins_pre);
        uint solt1 = DECODE_SLOT1(ins_pre);

        uint rowBase = row * NR_SLOTS * 4;

        ulong t0 = ht[rowBase + solt0 * 4];
        ulong t1 = ht[rowBase + solt1 * 4];

        uint x = t0 & 0xffffffff;
        uint y = t0 >> 32;
        uint z = t1 & 0xffffffff;
        uint w = t1 >> 32;

        values[j] = w;
        values[j - 1] = z;
        values[j - 2] = y;
        values[j - 3] = x;

        if (dup_to_watch == -1) {
            dup_to_watch = x;
            if (y == x || z == x || w == x)
                dup_value = 1;
        } else if (x == dup_to_watch || y == dup_to_watch 
                || z == dup_to_watch || w == dup_to_watch) {
            dup_value = 1;
        }

        if (!i)
           break;
        --i;
        j -= 4;
    } while(1);

    if (dup_value)
        return ;

    uint sol_i = *sols_nr;
    *sols_nr = sol_i + 1;
    if (sol_i >= MAX_SOLS)
        return ;
    for (uint i = 0; i < (1 << PARAM_K); ++i)
        sols->values[sol_i][i] = values[i];
}

__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void kernel_sols(__global ulong * restrict ht0,
        __global ulong * restrict ht1, 
        __global sols_t * restrict sols,
        __global uint * restrict rowCountersSrc)
{
    __global ulong  *htabs[2] = { ht0, ht1 };

    __local uint    rowCsSrc[1 << 17];
    __local uint    tuples[512][2];
    
    ulong   rowHash[NR_SLOTS];

    uint    cnt;
    bool    load = 1;
    uint    load_num;
    uint    load_i, load_j;
    ulong   mask = 0xffffff00000000;
    uint    tups_num = 0;
    bool    store;
    uint    collNum = 0;
    
    #pragma unroll 16
    for (uint i = 0; i < (1 << 17); ++i)
        rowCsSrc[i] = rowCountersSrc[i];

    #pragma ivdep array(rowHash)
    for (uint tid = 0; tid < (1 << 20); ) {
        if (load) {
            uint rowIdx = tid >> 3;
            uint rowOffset = BITS_PER_ROW * (tid & 0x7);
            cnt = (rowCsSrc[rowIdx] >> rowOffset) & ROW_MASK;
            if (cnt > NR_ROWS)
                cnt = NR_ROWS;
            load_num = 0;
            load_i = 0;
            load_j = 1;
            load = 0;
            store = 1;
        }

        if (cnt > 1) {
            if (load_num < cnt) {
                uint htOffset = (tid * NR_SLOTS + load_num) * 4 + 3;
                rowHash[load_num++] = ht0[htOffset];
            }

            if (load_num > 1 && load_i < cnt -1) {
                ulong x = rowHash[load_i];
                ulong y = rowHash[load_j];
                if ((x & mask) == (y & mask) && store) {
                    tuples[tups_num][0] = x & 0xffffffff;
                    tuples[tups_num][1] = y & 0xffffffff;
                    ++tups_num;
                    store = 0;
                    ++collNum;  
                }
                if (load_j == cnt - 1) {
                    ++load_i;
                    load_j = load_i + 1;
                } else {
                    ++load_j;
                }
            }
            if (load_i == cnt - 1) {
                load = 1;
                ++tid;
            }
        } else {
            load = 1;
            ++tid;
        }
    }

    uint sols_nr = 0;
    for (uint i = 0; i < tups_num; ++i)
        potential_sol(htabs, sols, tuples[i][0], tuples[i][1], &sols_nr);
    sols->nr = sols_nr;
}



