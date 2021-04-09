### Open Computing Language (OpenCl)


### Installation
Add the following repo to /etc/yum/repos.d/opencl.repo:
```
[copr:copr.fedorainfracloud.org:jdanecki:intel-opencl]
name=Copr repo for intel-opencl owned by jdanecki
baseurl=https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/fedora-$releasever-$basearch/
type=rpm-md
skip_if_unavailable=True
gpgcheck=1
gpgkey=https://download.copr.fedorainfracloud.org/results/jdanecki/intel-opencl/pubkey.gpg
repo_gpgcheck=0
enabled=1
enabled_metadata=1
```

```console
$ sudo dnf install opencl-headers intel-opencl ocl-icd-devel
```

After this we will be able to link our programs with  /usr/lib64/libOpenCL.so 
using -lopencl.
This library will read the files in `/etc/OpenCL/vendors/*.icd` which specifies
the library for the selected platform. `icd` stands for Installable Client
Driver:
```console
$ ls /etc/OpenCL/vendors/
intel.icd mesa.icd pocl.icd
$ cat /etc/OpenCL/vendors/intel.icd 
/usr/lib64/intel-opencl/libigdrcl.so
```

### Programming model
Parts of program run on the CPU which is called the host, and another part is
run on the GPU which is called the device. 
If we recall from the [architecture section](../README.md#architecture) we
mentioned that there are different types of systems. The CPU and GPU can share
the same memory or have separate memory. We need to set everything up in the
host code with regards to memory, like allocate memory for the device and also
copy data over to that memory. This is done with the OpenCL API. 

#### Workgroups


### mesa
https://www.mesa3d.org/ (I think)
 
### pocl
http://portablecl.org/

### clinfo
Is a command line tool for getting information a OpenCL platforms.
```console
$ sudo dnf install -y clinfo
```
