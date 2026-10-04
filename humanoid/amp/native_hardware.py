"""Typed native hardware readback, not platform or motion-quality admission.

The selected Gradmotion resource must separately be verified from the actual
task API as ESKU000001 / one GPU.  A CUDA-reported ``RTX 4090`` or ``RTX 4090 D``
name does not identify the physical chip's D/non-D variant.  This record only
checks the observed runtime family, memory, count and compute capability; it
never creates a simulator, executes a policy, fits a model or issues a smoke
certificate.  Supplying torch as an argument preserves Isaac Gym import order.
"""


SCHEMA = 'head_native_runtime_hardware_v1'
GIB = 1024 ** 3
MIN_MEMORY_BYTES = 23 * GIB
MAX_MEMORY_BYTES = 25 * GIB
RECORD_KEYS = frozenset(('schema', 'platform', 'cuda_available',
    'cuda_device_count', 'devices', 'torch_version', 'torch_cuda_version'))
DEVICE_KEYS = frozenset(('index', 'name', 'total_memory_bytes',
                         'compute_capability'))
RUNTIME_NAMES = frozenset(('NVIDIAGEFORCERTX4090', 'NVIDIAGEFORCERTX4090D'))


def read_hardware(torch, platform_name):
    """Read actual torch fields without simulation, coercion or SKU inference.

    Unavailable CUDA never triggers device-name/property queries. Unexpected
    count/flag types are retained for the strict validator, not converted into
    a plausible single-GPU record. Multiple visible devices remain recorded.
    """
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    devices = []
    if available is True and type(count) is int and count >= 0:
        for index in range(count):
            name = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            devices.append(dict(index=index, name=name,
                total_memory_bytes=props.total_memory,
                compute_capability=[props.major, props.minor]))
    version = torch.__version__
    # PyTorch's TorchVersion is a str subclass. Keep its actual textual value;
    # do not stringify non-string malformed inputs into admissible metadata.
    if isinstance(version, str):
        version = str(version)
    cuda_version = torch.version.cuda
    if isinstance(cuda_version, str):
        cuda_version = str(cuda_version)
    return dict(schema=SCHEMA, platform=platform_name, cuda_available=available,
        cuda_device_count=count, devices=devices, torch_version=version,
        torch_cuda_version=cuda_version)


def _fail(message):
    # Field-specific diagnostics never echo arbitrary inputs or environment data.
    raise ValueError('Native head hardware: ' + message)


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != expected:
        _fail('missing/extra or untyped ' + label + ' fields')


def validate_hardware_record(record):
    """Validate the exact runtime contract, without mutating the received record.

    Success returns True. This does NOT verify a platform SKU, distinguish chip
    variants, authorize fitting, certify native execution or establish quality.
    The only name normalization removes ASCII spaces and uppercases ASCII text;
    suffixes, prefixes, laptop variants, other cards and control bytes fail.
    """
    _keys(record, RECORD_KEYS, 'record')
    if type(record['schema']) is not str or record['schema'] != SCHEMA:
        _fail('wrong schema')
    if type(record['platform']) is not str or record['platform'] != 'linux':
        _fail('Linux runtime required')
    if type(record['cuda_available']) is not bool or record['cuda_available'] is not True:
        _fail('actual available CUDA boolean required')
    if type(record['cuda_device_count']) is not int or record['cuda_device_count'] != 1:
        _fail('exactly one typed CUDA device required')
    for key in ('torch_version', 'torch_cuda_version'):
        if type(record[key]) is not str or not record[key].strip():
            _fail('missing/untyped ' + key)
    devices = record['devices']
    if type(devices) is not list or len(devices) != 1:
        _fail('exactly one actual device record required')
    device = devices[0]
    _keys(device, DEVICE_KEYS, 'device')
    if type(device['index']) is not int or device['index'] != 0:
        _fail('actual CUDA index zero required')
    name = device['name']
    if (type(name) is not str or not name.isascii()
            or name.replace(' ', '').upper() not in RUNTIME_NAMES):
        _fail('unsupported exact runtime device name')
    memory = device['total_memory_bytes']
    if type(memory) is not int or not MIN_MEMORY_BYTES <= memory <= MAX_MEMORY_BYTES:
        _fail('typed device memory must be within 23 to 25 GiB')
    capability = device['compute_capability']
    if (type(capability) is not list or len(capability) != 2
            or any(type(value) is not int for value in capability)
            or capability != [8, 9]):
        _fail('typed compute capability 8.9 required')
    return True
