# utility script to profile per-line GPU memory usage
# modify this if you're using it for a different project
# to use this script, add the following lines at the top of your code:
# import sys; from gpu_profile import gpu_profile; sys.settrace(gpu_profile)

import linecache
import os

import torch
from py3nvml import py3nvml

print_tensor_sizes = False

logfile = "memory_log.txt"

if os.path.exists(logfile):
    os.remove(logfile)

## Global variables
last_tensor_sizes = set()
last_meminfo_used = 0
lineno = None
func_name = None
filename = None
module_name = None


def gpu_profile(frame, event, arg):
    # it is _about to_ execute (!)
    global last_tensor_sizes
    global last_meminfo_used
    global lineno, func_name, filename, module_name

    if event == "line":
        try:
            # about _previous_ line (!)
            if lineno is not None:
                py3nvml.nvmlInit()
                handle = py3nvml.nvmlDeviceGetHandleByIndex(
                    int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
                )
                meminfo = py3nvml.nvmlDeviceGetMemoryInfo(handle)
                line = linecache.getline(filename, lineno - 1)
                where_str = module_name + " " + func_name + ":" + str(lineno)

                new_meminfo_used = meminfo.used
                mem_increment = new_meminfo_used - last_meminfo_used
                with open(logfile, "a+") as f:
                    allowed_funcs = ["staged_speculative_decoding"]
                    if func_name in allowed_funcs and mem_increment != 0:
                        f.write(
                            f"{where_str:<50}"
                            f":{(mem_increment)/1024**2:<7.1f}Mb "
                            f":{(new_meminfo_used)/1024**2:<7.1f}Mb "
                            f"{line.rstrip()}\n"
                        )

                    last_meminfo_used = new_meminfo_used
                    if print_tensor_sizes is True:
                        for tensor in get_tensors():
                            if not hasattr(tensor, "dbg_alloc_where"):
                                tensor.dbg_alloc_where = where_str
                        new_tensor_sizes = {
                            (type(x), tuple(x.size()), x.dbg_alloc_where)
                            for x in get_tensors()
                        }
                        for t, s, loc in new_tensor_sizes - last_tensor_sizes:
                            f.write(f"+ {loc:<50} {str(s):<20} {str(t):<10}\n")
                        for t, s, loc in last_tensor_sizes - new_tensor_sizes:
                            f.write(f"- {loc:<50} {str(s):<20} {str(t):<10}\n")
                        last_tensor_sizes = new_tensor_sizes
                py3nvml.nvmlShutdown()

            # save details about line _to be_ executed
            lineno = None

            func_name = frame.f_code.co_name
            filename = frame.f_globals["__file__"]
            if filename.endswith(".pyc") or filename.endswith(".pyo"):
                filename = filename[:-1]
            module_name = frame.f_globals["__name__"]
            lineno = frame.f_lineno

            # only profile code within the parent folder, otherwise there are too many
            # function calls into other pytorch scripts
            # need to modify the key words below to suit your case.
            if os.getcwd() not in os.path.abspath(filename):
                lineno = None  # skip current line evaluation

            if (
                "car_datasets" in filename
                or "_exec_config" in func_name
                or "gpu_profile" in module_name
                or "tee_stdout" in module_name
            ):
                lineno = None  # skip other unnecessary lines

            return gpu_profile

        except (KeyError, AttributeError):
            pass

    return gpu_profile


def get_tensors(gpu_only=True):
    import gc

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception as e:
            pass
