import cpuinfo
import platform
import psutil
from numba import cuda
 
def cpu_and_system_props():    
    props = {}    
    info = cpuinfo.get_cpu_info()
    un = platform.uname()
    props["cpu_name"] = info["brand_raw"]
    props["ram_size"] = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    props["os_name"] = f"{un.system} {un.release}"
    props["os_version"] = f"{un.version}"
    props["os_machine"] = f"{un.machine}"    
    return props    

def gpu_props():
    gpu = cuda.get_current_device()
    props = {}
    props["name"] = gpu.name.decode("ASCII")
    props["max_threads_per_block"] = gpu.MAX_THREADS_PER_BLOCK
    props["max_block_dim_x"] = gpu.MAX_BLOCK_DIM_X
    props["max_block_dim_y"] = gpu.MAX_BLOCK_DIM_Y
    props["max_block_dim_z"] = gpu.MAX_BLOCK_DIM_Z
    props["max_grid_dim_x"] = gpu.MAX_GRID_DIM_X
    props["max_grid_dim_y"] = gpu.MAX_GRID_DIM_Y
    props["max_grid_dim_z"] = gpu.MAX_GRID_DIM_Z    
    props["max_shared_memory_per_block"] = gpu.MAX_SHARED_MEMORY_PER_BLOCK
    props["async_engine_count"] = gpu.ASYNC_ENGINE_COUNT
    props["can_map_host_memory"] = gpu.CAN_MAP_HOST_MEMORY
    props["multiprocessor_count"] = gpu.MULTIPROCESSOR_COUNT
    props["warp_size"] = gpu.WARP_SIZE
    props["unified_addressing"] = gpu.UNIFIED_ADDRESSING
    props["pci_bus_id"] = gpu.PCI_BUS_ID
    props["pci_device_id"] = gpu.PCI_DEVICE_ID
    props["compute_capability"] = gpu.compute_capability            
    CC_CORES_PER_SM_DICT = {
        (2,0) : 32,
        (2,1) : 48,
        (3,0) : 256,
        (3,5) : 256,
        (3,7) : 256,
        (5,0) : 128,
        (5,2) : 128,
        (6,0) : 64,
        (6,1) : 128,
        (7,0) : 64,
        (7,5) : 64,
        (8,0) : 64,
        (8,6) : 128
        }
    props["cores_per_SM"] = CC_CORES_PER_SM_DICT.get(gpu.compute_capability)
    props["cores_total"] = props["cores_per_SM"] * gpu.MULTIPROCESSOR_COUNT
    return props 