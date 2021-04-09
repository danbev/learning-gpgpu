#include <stdio.h>
#include <stdlib.h>
//#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define VECTOR_SIZE 4

const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float a,                 \n"
"                  __global float* x,       \n"
"                  __global float* y) {     \n"
"    const int i = get_global_id(0);        \n"
"    y[i] += a * x[i];                      \n"
"}                                          \n";

int main(void) {
  int i;
  float alpha = 2.0;
  float* x = (float*) malloc(sizeof(float) * VECTOR_SIZE);
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  x[3] = 4.0;

  float* y = (float*) malloc(sizeof(float) * VECTOR_SIZE);
  y[0] = 0.0;
  y[1] = 0.0;
  y[2] = 0.0;
  y[3] = 0.0;

  // Get platform and device information
  cl_uint num_platforms;

  // There can be multiple platforms, like one for intel and another for
  // amd for example. Here we are only interested in the number of platforms
  // so we pass 0 which is the length of platform array.
  cl_int cl_status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (cl_status != CL_SUCCESS) {
    printf("clGetPlatformIDs failed to get the number of platforms!\n");
    exit(1);
  }
  printf("Found %d number of OpenCL platforms\n", num_platforms);

  cl_platform_id* platforms = NULL;
  platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id)*num_platforms);
  // Now we query for information about the platforms using the size
  // num_platforms:
  cl_status = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (cl_status != CL_SUCCESS) {
    printf("clGetPlatformIDs failed to get info about the platforms!\n");
    exit(1);
  }

  for (int i = 0; i < num_platforms; i++) {
    char buf[1024];
    size_t actual_size;
    cl_status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 1024, &buf, &actual_size);
    if (cl_status != CL_SUCCESS) {
      printf("clGetPlatformInfo failed to get info for platform %d!\n", i);
      exit(1);
    }
    printf("%d Platform name: %s\n", i, buf);
  }
  printf("Using platform 0\n");

  cl_uint num_devices;
  // Query the number of devices that platform 0 has of type CL_DEVICE_TYPE_GPU 
  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (cl_status != CL_SUCCESS) {
    printf("clGetDeviceIDs failed to get number for devices\n");
    exit(1);
  }
  printf("Number of devices: %d\n", num_devices);

  // Now we query for the devices them selves using the retrieved num_devices
  // from the previous call.
  cl_device_id* device_list = NULL;
  device_list = (cl_device_id*) malloc(sizeof(cl_device_id)*num_devices);
  cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
  if (cl_status != CL_SUCCESS) {
    printf("clGetDeviceIDs failed to get info for devices\n");
    exit(1);
  }
  for (int i = 0; i < num_devices; i++) {
    char buf[1024];
    size_t actual_size;
    cl_status = clGetDeviceInfo(device_list[i], CL_DEVICE_NAME , 1024, &buf, &actual_size);
    if (cl_status != CL_SUCCESS) {
      printf("clGetDeviceInfo failed to get CL_DEVICE_NAME for device %d!\n", i);
      exit(1);
    }
    printf("Device name: %s\n", buf);
    int compute_units;
    cl_status = clGetDeviceInfo(device_list[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &compute_units, NULL);
    if (cl_status != CL_SUCCESS) {
      printf("clGetDeviceInfo failed to get CL_DEVICE_MAX_COMPUTE_UNITS for device %d!\n", i);
      exit(1);
    }
    printf("Compute units: %d\n", compute_units);
  }

  // A context is used for managing command queues, memory, programs and
  // kernel objects.
  cl_context context;
  cl_context_properties* props = NULL;
  void (*callback)(const char* errInfo, const void* private_info, size_t cb, void* user_data) = NULL;
  void* user_data = NULL;
  context = clCreateContext(props, num_devices, device_list, callback , user_data, &cl_status);
  if (context == NULL) {
      printf("clCreateContext failed\n");
      exit(1);
  }

  cl_command_queue_properties queue_props = 0;
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], queue_props, &cl_status);

  size_t size = VECTOR_SIZE * sizeof(float);
  void* host_ptr = NULL;
  cl_mem x_d = clCreateBuffer(context, CL_MEM_READ_ONLY, size, host_ptr, &cl_status);
  if (x_d == NULL) {
      printf("clCreateBuffer failed\n");
      exit(1);
  }
  cl_mem y_d = clCreateBuffer(context, CL_MEM_READ_ONLY, size, host_ptr, &cl_status);
  if (y_d == NULL) {
      printf("clCreateBuffer failed\n");
      exit(1);
  }

  cl_bool blocking_write = CL_TRUE;
  size_t offset = 0;
  // The following call will enque a command to write from the host buffer to
  // the device buffer (x_h -> x_d)
  cl_status = clEnqueueWriteBuffer(command_queue, x_d, CL_TRUE, 0, size, x, 0, NULL, NULL);
  if (cl_status != CL_SUCCESS) {
    printf("clEnqueueWriteBuffer failed to enqueue a write for x_d\n");
    exit(1);
  }
  cl_status = clEnqueueWriteBuffer(command_queue, y_d, CL_TRUE, 0, size, y, 0, NULL, NULL);
  if (cl_status != CL_SUCCESS) {
    printf("clEnqueueWriteBuffer failed to enqueue a write for y_d\n");
    exit(1);
  }

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &cl_status);
  if (program == NULL) {
      printf("clCreateProgramWithSource failed\n");
      exit(1);
  }

  // Compiles and links the program
  const char* options = NULL;
  void (*cb)(cl_program, void* user_data) = NULL;
  cl_status = clBuildProgram(program, 1, device_list, options, cb, user_data);
  if (cl_status != CL_SUCCESS) {
    printf("clBuildProgram failed to compile and link the program\n");
    exit(1);
  }

  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &cl_status);
  if (kernel == NULL) {
      printf("clCreateKernel failed\n");
      exit(1);
  }

  cl_status = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  if (cl_status != CL_SUCCESS) {
    printf("clSetKernelArg failed to set alpha argument\n");
    exit(1);
  }
  cl_status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x_d);
  if (cl_status != CL_SUCCESS) {
    printf("clSetKernelArg failed to set x argument\n");
    exit(1);
  }
  cl_status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y_d);
  if (cl_status != CL_SUCCESS) {
    printf("clSetKernelArg failed to set y argument\n");
    exit(1);
  }

  size_t work_dim = 1;              // 1, 2, or 3 dimensions are supported
  size_t global_size = VECTOR_SIZE; // the number of work-items to execute
  size_t local_size = 4;            // the number of work-items to group into a work-group
  // NDRange for n-dimension range perhaps?
  //    []    []       []     []          global size = 4
  // group0  group1  group2 group3        local_size  = 4
  cl_status = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, &global_size, &local_size, 0, NULL, NULL);
  if (cl_status != CL_SUCCESS) {
    printf("clEnqueueNDRangeKernel failed\n");
    exit(1);
  }

  // Read the cl memory y_clmem on device to the host variable y
  cl_status = clEnqueueReadBuffer(command_queue, y_d, CL_TRUE, 0, size, y, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  cl_status = clFlush(command_queue);
  cl_status = clFinish(command_queue);

  // Display the result to the screen
  for(i = 0; i < VECTOR_SIZE; i++)
    printf("%f, %f\n", x[i], y[i]);

  // Finally release all OpenCL allocated objects and host buffers.
  cl_status = clReleaseKernel(kernel);
  cl_status = clReleaseProgram(program);
  cl_status = clReleaseMemObject(x_d);
  cl_status = clReleaseMemObject(y_d);
  cl_status = clReleaseCommandQueue(command_queue);
  cl_status = clReleaseContext(context);
  free(x);
  free(y);
  free(platforms);
  free(device_list);
  return 0;
}
