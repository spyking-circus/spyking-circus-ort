import pyopencl

def get_first_gpu_device(platform_name):
    '''
    Get first gpu device on a platform that starts with substring
    '''
    ret = None
    for platform in pyopencl.get_platforms():
        if platform.name.startswith(platform_name):           
            for device in platform.get_devices():
                if pyopencl.device_type.to_string(device.type) == 'GPU':
                    ret = device
                    break
    return ret