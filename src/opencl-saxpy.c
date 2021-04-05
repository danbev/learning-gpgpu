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

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &cl_status);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &cl_status);

  // Create memory buffers on the device for each vector
  cl_mem x_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(float), NULL, &cl_status);
  cl_mem y_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(float), NULL, &cl_status);

  cl_status = clEnqueueWriteBuffer(command_queue, x_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), x, 0, NULL, NULL);
  cl_status = clEnqueueWriteBuffer(command_queue, y_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), y, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &cl_status);

  // Build the program
  cl_status = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &cl_status);

  cl_status = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  cl_status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x_clmem);
  cl_status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y_clmem);

  // Execute the OpenCL kernel on the list
  size_t global_size = VECTOR_SIZE; // Process the entire lists
  size_t local_size = 4;            // Process one item at a time
  cl_status = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Read the cl memory y_clmem on device to the host variable y
  cl_status = clEnqueueReadBuffer(command_queue, y_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), y, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  cl_status = clFlush(command_queue);
  cl_status = clFinish(command_queue);

  // Display the result to the screen
  for(i = 0; i < VECTOR_SIZE; i++)
    printf("%f, %f\n", x[i], y[i]);

  // Finally release all OpenCL allocated objects and host buffers.
  cl_status = clReleaseKernel(kernel);
  cl_status = clReleaseProgram(program);
  cl_status = clReleaseMemObject(x_clmem);
  cl_status = clReleaseMemObject(y_clmem);
  cl_status = clReleaseCommandQueue(command_queue);
  cl_status = clReleaseContext(context);
  free(x);
  free(y);
  free(platforms);
  free(device_list);
  return 0;
}
