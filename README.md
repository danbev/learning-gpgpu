### Learning General Purpose GPU
This project contains notes and code related to gpgpu.

### Architecture
```
+---------------------------------------------------------------------------+
|                          GPU                                              |
| +---------------------------------------------------------------+         |
| |                     Core 0                                    |         | 
| | +------------------------------------------------------+      |         |
| | |    Fetch + Decode instruction from GPU device memory |----+ |         |
| | +------------------------------------------------------+    | |         |
| |     ↓              ↓            ↓            ↓              | |         |
| | +--------+    +---------+   +---------+                     | |         |
| | |  ALU   |    |  ALU    |   | ALU     |     ...             | |         |
| | +--------+    +---------+   +---------+                     | |   ...   |
| | +--------+    +---------+   +---------+                     | |         |
| | |Register|    |Registers|   |Registers|                     | |         |
| | +--------+    +---------+   +---------+                     | |         |
| |                                                             | |         |
| | +------------------------------------------------------+    | |         |
| | |       GPU Memory                                     |<---+ |         |
| | |                                                      |      |         |
| | |                                                      |      |         |
| | |                                                      |      |         |
| | +------------------------------------------------------+      |         |
| +---------------------------------------------------------------+         |
+--------------------------------==-----------------------------------------+
```
So the GPU will read instructions from its memory and decode them. Then it will
pass the instrution to all the ALU's, so they will all be passed the same
instruction. But each ALU has registers that are separate from each other, so
the values that the instruction operated on can be different (Single Instruction
Multiple Data).

_work_in_progress_


### Programming models
* [OpenCL](./notes/opencl.md)

### CPU support
```console
$ lspci | grep Graph
00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620 (rev 07)
```
So my machine has an integrated graphics (controller) which means that the
CPU and GPU are on the same chip.
```console
$ clinfo
  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Gen9 HD Graphics NEO
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 2.1 NEO 
  Driver Version                                  20.28.17293
  Device OpenCL C Version                         OpenCL C 2.0 
  Device Type                                     GPU
  Device Profile                                  FULL_PROFILE
  Device Available                                Yes
  Compiler Available                              Yes
  Linker Available                                Yes
  Max compute units                               24
  Max clock frequency                             1150MHz
  Device Partition                                (core)
    Max number of sub-devices                     0
    Supported partition types                     None
    Supported affinity domains                    (n/a)
  Max work item dimensions                        3
  Max work item sizes                             256x256x256
  Max work group size                             256
  Preferred work group size multiple              32
  Max sub-groups per work group                   32
  Sub-group sizes (Intel)                         8, 16, 32
```
The one I've got has 24 Execution/Compute units. 

