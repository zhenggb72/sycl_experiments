#include <level_zero/ze_api.h>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <numeric>
#include <string>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <mpi.h>

#define zeCall(myZeCall)                               \
    do {                                               \
        ze_result_t result = (myZeCall);            \
        if (result != ZE_RESULT_SUCCESS) {           \
            std::cout << "Error " << std::hex << result << std::dec << " at "                   \
                      << #myZeCall << ": "             \
                      << __FUNCTION__ << ": "          \
                      << std::dec << __LINE__ << "\n"; \
            std::terminate();                          \
        }                                              \
    } while (0);

ze_driver_handle_t driver;
std::vector<ze_device_handle_t> devices;

ze_context_handle_t context;
ze_device_handle_t device;

static void copy_content(void *sptr, void *dptr, unsigned long size)
{
    ze_result_t ret = ZE_RESULT_SUCCESS;

    ze_command_queue_handle_t cmdQueue;
    ze_command_queue_desc_t cmdQueueDesc = { ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, NULL };
    cmdQueueDesc.index = 0;
    cmdQueueDesc.flags = 0;
    cmdQueueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    cmdQueueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    uint32_t numQueueGroups = 0;
    ret = zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, NULL);
    assert(ret == ZE_RESULT_SUCCESS && numQueueGroups);
    ze_command_queue_group_properties_t *queueProperties = (ze_command_queue_group_properties_t *) malloc(sizeof(ze_command_queue_group_properties_t) * numQueueGroups);
    ret = zeDeviceGetCommandQueueGroupProperties(device, &numQueueGroups, queueProperties);
    cmdQueueDesc.ordinal = -1;
    for (int i = 0; i < numQueueGroups; i++) {
        if (queueProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            cmdQueueDesc.ordinal = i;
            break;
        }
    }
    assert(cmdQueueDesc.ordinal != -1);
    ret = zeCommandQueueCreate(context, device, &cmdQueueDesc, &cmdQueue);
    assert(ret == ZE_RESULT_SUCCESS);
    free(queueProperties);

    ze_command_list_handle_t cmdList;
    ze_command_list_desc_t cmdListDesc = { ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, NULL, cmdQueueDesc.ordinal, 0 };
    ret = zeCommandListCreate(context, device, &cmdListDesc, &cmdList);
    assert(ret == ZE_RESULT_SUCCESS);

    ret = zeCommandListAppendMemoryCopy(cmdList, dptr, sptr, size, NULL, 0, NULL);
    assert(ret == ZE_RESULT_SUCCESS);
    ret = zeCommandListClose(cmdList);
    assert(ret == ZE_RESULT_SUCCESS);
    ret = zeCommandQueueExecuteCommandLists(cmdQueue, 1, &cmdList, NULL);
    assert(ret == ZE_RESULT_SUCCESS);

    ret = zeCommandQueueSynchronize(cmdQueue, UINT32_MAX);
    assert(ret == ZE_RESULT_SUCCESS);
    ret = zeCommandListDestroy(cmdList);
    assert(ret == ZE_RESULT_SUCCESS);

    ret = zeCommandQueueDestroy(cmdQueue);
    assert(ret == ZE_RESULT_SUCCESS);
}

static int pidfd_open(pid_t pid, unsigned int flags)
{
        return syscall(__NR_pidfd_open, pid, flags);
}

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char name[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(name, &len);

    printf("Hello host %s, rank %d, size %d\n", name, rank, size);

    zeInit(ZE_INIT_FLAG_GPU_ONLY);

    uint32_t driver_count = 0;
    zeCall(zeDriverGet(&driver_count, nullptr));
    zeCall(zeDriverGet(&driver_count, &driver));

    uint32_t device_count = 0;
    zeCall(zeDeviceGet(driver, &device_count, nullptr));
    devices.resize(device_count);
    zeCall(zeDeviceGet(driver, &device_count, devices.data()));

    device = devices[rank];

    ze_context_desc_t context_desc{};
    zeCall(zeContextCreate(driver, &context_desc, &context));

    //  malloc buffer
    ze_external_memory_export_desc_t export_desc = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        //ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
    };
    ze_device_mem_alloc_desc_t alloc_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = &export_desc,
        .flags = 0,
        .ordinal = 0,   /* We currently support a single memory type */
    };

    size_t alloc_size = 128 * 1024 * 1024;
    size_t alignment = 1;
    void *ptr;
    zeMemAllocDevice(context, &alloc_desc, alloc_size, alignment, device, &ptr);

    // init buffer
    int *host_ptr;
    ze_host_mem_alloc_desc_t host_desc = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, NULL };
    host_desc.flags = 0;
    zeMemAllocHost(context, &host_desc, alloc_size, alignment, (void **)&host_ptr);
    for (int i = 0; i< alloc_size / sizeof(int); i++) {
        host_ptr[i] = i + rank;
    }
    copy_content(host_ptr, ptr, alloc_size);

    // immediately copy back and validate
    memset(host_ptr, 0, alloc_size);
    copy_content(ptr, host_ptr, alloc_size);

    for (int i = 0; i< alloc_size / sizeof(int); i++) {
        if (host_ptr[i] != i + rank)
            fprintf(stderr, "error %d %d \n", host_ptr[i], i + rank);
    }
    fprintf(stderr, "[%d] self validation done\n", rank);

    // export IPC handle
    int fd;
    ze_external_memory_export_fd_t export_fd = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        0 // [out] fd
    };

    // Link the export request into the query
    ze_memory_allocation_properties_t alloc_props {};
    alloc_props.pNext = &export_fd;
    zeMemGetAllocProperties(context, ptr, &alloc_props, nullptr);
    fd = export_fd.fd;
    fprintf(stderr, "[%d] export to fd: %d\n", rank, fd);

    // allgather
    int fds[2];
    MPI_Allgather(&fd, 1, MPI_INT, fds, 1, MPI_INT, MPI_COMM_WORLD);

    int pid = getpid();
    int pids[2];
    MPI_Allgather(&pid, 1, MPI_INT, pids, 1, MPI_INT, MPI_COMM_WORLD);

    int peer_rank = rank ^ 1;

    int pid_fd = pidfd_open(pids[peer_rank], 0);
    int peer_fd = syscall(438, pid_fd, fds[peer_rank], 0);
    fprintf(stderr, "[%d] get peer fd: %d\n", rank, peer_fd);

#if 1
    // create physical memory
    ze_external_memory_import_fd_t import_fd = { 
        .stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
        .pNext = nullptr,
        .flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        //.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        .fd = peer_fd };

    ze_physical_mem_desc_t physical_alloc_desc = {
        .stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC, 
        .pNext = &import_fd, 
        .flags = ZE_PHYSICAL_MEM_FLAG_ALLOCATE_ON_DEVICE, 
        .size = alloc_size
    };
    fprintf(stderr, "[%d] zePhysicalMemCreate: fd: %d size: %ld \n", rank, peer_fd, alloc_size);

    ze_physical_mem_handle_t hPhysicalMemory;
    zeCall(zePhysicalMemCreate(context, device, &physical_alloc_desc, &hPhysicalMemory));

    void *mapped_ptr;
    zeCall(zeVirtualMemReserve(context, nullptr, alloc_size, &mapped_ptr));

    zeCall(zeVirtualMemMap(context, mapped_ptr, alloc_size, hPhysicalMemory, 0, ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE));
 
    ze_ipc_mem_handle_t ipc_handle;
    zeCall(zeMemGetIpcHandle(context, mapped_ptr, &ipc_handle));
    void *peer_ptr;
    zeCall(zeMemOpenIpcHandle(context, device, ipc_handle, 0, &peer_ptr));
    fprintf(stderr, "[%d] open handle from %p to %p \n", rank, mapped_ptr, peer_ptr);
#else
    // Set up the request to import the external memory handle
    ze_external_memory_import_fd_t import_fd = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
        //ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
        peer_fd
    };

    void *peer_ptr;
    ze_device_mem_alloc_desc_t alloc_desc2 = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = &import_fd,
        .flags = 0,
        .ordinal = 0,   /* We currently support a single memory type */
    };
    zeCall(zeMemAllocDevice(context, &alloc_desc2, alloc_size, alignment, device, &peer_ptr));
    fprintf(stderr, "[%d] malloc device from fd to %p \n", rank, peer_ptr);
#endif

    copy_content(peer_ptr, host_ptr, alloc_size);

    int count = 0;
    for (int i = 0; i< alloc_size / sizeof(int); i++) {
        if (host_ptr[i] != i + peer_rank && count < 10) {
            fprintf(stderr, "[%d] validation error got: %d expected: %d \n", rank, host_ptr[i], i + peer_rank);
            count ++;
        }
    }
    
    MPI_Finalize();
}

