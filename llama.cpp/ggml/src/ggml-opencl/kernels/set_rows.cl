#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK8_0 32
typedef struct {
    half d;
    char qs[QK8_0];
} block_q8_0;

// Match ggml quantize_row_q8_0-style: per-block absmax scale, int8 quants
inline void kernel_set_rows_quantize_block_q8_0(global const float * src_block, global block_q8_0 * dst_blk) {
    float amax = 0.f;
    for (int j = 0; j < QK8_0; j++) {
        float av = fabs(src_block[j]);
        amax = max(amax, av);
    }
    float d = amax / 127.f;
    float id = d > 0.f ? (1.f / d) : 0.f;
    dst_blk->d = (half)d;
    for (int j = 0; j < QK8_0; j++) {
        float x0 = src_block[j] * id;
        dst_blk->qs[j] = convert_char_sat((int)round(x0));
    }
}

// Match ggml quantize_row_q4_0_ref: asymmetric 4-bit, d from signed element of max |x|
#define QK4_0_KV 32
typedef struct {
    half d;
    uchar qs[QK4_0_KV / 2];
} block_q4_0;

// int32 -> low 8 bits reinterpreted as signed int8 (two's complement)
inline char kernel_set_rows_i32_as_int8_truncate(int iq) {
    int low8 = iq & 255;
    if (low8 >= 128) {
        return (char)(low8 - 256);
    }
    return (char)low8;
}

inline uchar kernel_set_rows_q4_packed_nibble_ref(float xv) {
    int iq = convert_int(xv + 8.5f);
    char c = kernel_set_rows_i32_as_int8_truncate(iq);
    int m = c <= 15 ? (int) c : 15;
    return convert_uchar(m);
}

inline void kernel_set_rows_quantize_block_q4_0(global const float * src_block, global block_q4_0 * dst_blk) {
    float amax = 0.f;
    float max_val = 0.f;
    for (int j = 0; j < QK4_0_KV; j++) {
        float v = src_block[j];
        float av = fabs(v);
        if (amax < av) {
            amax = av;
            max_val = v;
        }
    }
    float d = max_val / -8.f;
    float id = d != 0.f ? (1.f / d) : 0.f;
    dst_blk->d = (half)d;
    for (int j = 0; j < QK4_0_KV / 2; ++j) {
        float x0 = src_block[0               + j] * id;
        float x1 = src_block[QK4_0_KV / 2 + j] * id;
        uchar xi0 = kernel_set_rows_q4_packed_nibble_ref(x0);
        uchar xi1 = kernel_set_rows_q4_packed_nibble_ref(x1);
        dst_blk->qs[j] = xi0 | (xi1 << 4);
    }
}

// v = { mp, L, d }
inline uint fastdiv(uint n, uint4 v) {
    uint msbs;
    msbs = mul_hi(n, v.s0);
    return (msbs + n) >> v.s1;
}
inline uint fastmod(uint n, uint4 v) {
    uint q = fastdiv(n, v);
    return n - q * v.s2;
}

kernel void kernel_set_rows_f32_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_q8_0_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global block_q8_0 * dst_row = (global block_q8_0 *)(dst + i1*nb1 + i02*nb2 + i03*nb3);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        kernel_set_rows_quantize_block_q8_0(src_row + ind * QK8_0, dst_row + ind);
    }
}

kernel void kernel_set_rows_q8_0_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global block_q8_0 * dst_row = (global block_q8_0 *)(dst + i1*nb1 + i02*nb2 + i03*nb3);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        kernel_set_rows_quantize_block_q8_0(src_row + ind * QK8_0, dst_row + ind);
    }
}

kernel void kernel_set_rows_q4_0_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global block_q4_0 * dst_row = (global block_q4_0 *)(dst + i1*nb1 + i02*nb2 + i03*nb3);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        kernel_set_rows_quantize_block_q4_0(src_row + ind * QK4_0_KV, dst_row + ind);
    }
}

kernel void kernel_set_rows_q4_0_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global block_q4_0 * dst_row = (global block_q4_0 *)(dst + i1*nb1 + i02*nb2 + i03*nb3);
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        kernel_set_rows_quantize_block_q4_0(src_row + ind * QK4_0_KV, dst_row + ind);
    }
}
kernel void kernel_set_rows_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}
