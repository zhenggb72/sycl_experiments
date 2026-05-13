__kernel void reduce_scatter_kernel_2( \
    __global long* peer_buf_0,
    __global long* peer_buf_1,
    __global long* local_buf,
    const long count,
    const int is_write,
    const int rank) {

    const size_t thread_id = get_global_id(0);
    if (is_write) {
        long data = local_buf[thread_id];
        peer_buf_0[rank * count + thread_id] = data;
        peer_buf_1[rank * count + thread_id] = data;
    }
    else {
        local_buf[0 * count + thread_id] = peer_buf_0[thread_id];
        local_buf[1 * count + thread_id] = peer_buf_1[thread_id];
    }
}

__kernel void reduce_scatter_kernel_4( \
    __global long* peer_buf_0,
    __global long* peer_buf_1,
    __global long* peer_buf_2,
    __global long* peer_buf_3,
    __global long* local_buf,
    const long count,
    const int is_write,
    const int rank) {

    const size_t thread_id = get_global_id(0);
    if (is_write) {
        long data = local_buf[thread_id];
        peer_buf_0[rank * count + thread_id] = data;
        peer_buf_1[rank * count + thread_id] = data;
        peer_buf_2[rank * count + thread_id] = data;
        peer_buf_3[rank * count + thread_id] = data;
    }
    else {
        local_buf[0 * count + thread_id] = peer_buf_0[thread_id];
        local_buf[1 * count + thread_id] = peer_buf_1[thread_id];
        local_buf[2 * count + thread_id] = peer_buf_2[thread_id];
        local_buf[3 * count + thread_id] = peer_buf_3[thread_id];
    }
}

