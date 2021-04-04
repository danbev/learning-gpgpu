#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define VECTOR_SIZE 4

const char *saxpy_kernel =
"__kernel                                   \n"
"void saxpy_kernel(float a,                 \n"
"                  __global float *x,       \n"
"                  __global float *y,       \n"
"                  __global float *z) {     \n"
"    //Get the index of the work-item       \n"
"    int index = get_global_id(0);          \n"
"    z[index] = alpha * x[index] + y[index]; \n"
"}                                          \n";

int main(void) {
  int i;
  // Allocate space for vectors A, B and C
  float alpha = 2.0;
  float* x = (float*) malloc(sizeof(float) * VECTOR_SIZE);
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  x[3] = 4.0;
  float* y = (float*) malloc(sizeof(float) * VECTOR_SIZE);
  float* z = (float*) malloc(sizeof(float) * VECTOR_SIZE);
  for(i = 0; i <= VECTOR_SIZE; i++)
  {
    x[i] = 1.0;
    y[i] = 0.0;
  }

  // Get platform and device information
  cl_platform_id * platforms = NULL;
  cl_uint     num_platforms;
  //Set up the Platform
  cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)
  malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);

  //Get the devices list and choose the device you want to run on
  cl_device_id     *device_list = NULL;
  cl_uint           num_devices;

  clStatus = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
  device_list = (cl_device_id *) 
  malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs( platforms[0],CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext( NULL, num_devices, device_list, NULL, NULL, &clStatus);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_list[0], 0, &clStatus);

  // Create memory buffers on the device for each vector
  cl_mem x_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem y_clmem = clCreateBuffer(context, CL_MEM_READ_ONLY,VECTOR_SIZE * sizeof(float), NULL, &clStatus);
  cl_mem z_clmem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,VECTOR_SIZE * sizeof(float), NULL, &clStatus);

  clStatus = clEnqueueWriteBuffer(command_queue, x_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), x, 0, NULL, NULL);
  clStatus = clEnqueueWriteBuffer(command_queue, y_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), y, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1,(const char **)&saxpy_kernel, NULL, &clStatus);

  // Build the program
  clStatus = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "saxpy_kernel", &clStatus);

  // Set the arguments of the kernel
  clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&alpha);
  clStatus = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x_clmem);
  clStatus = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y_clmem);
  clStatus = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&z_clmem);

  // Execute the OpenCL kernel on the list
  size_t global_size = VECTOR_SIZE; // Process the entire lists
  size_t local_size = 64;           // Process one item at a time
  clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

  // Read the cl memory z_clmem on device to the host variable C
  clStatus = clEnqueueReadBuffer(command_queue, z_clmem, CL_TRUE, 0, VECTOR_SIZE * sizeof(float), z, 0, NULL, NULL);

  // Clean up and wait for all the comands to complete.
  clStatus = clFlush(command_queue);
  clStatus = clFinish(command_queue);

  // Display the result to the screen
  for(i = 0; i < VECTOR_SIZE; i++)
    printf("%f\n", z[i]);

  // Finally release all OpenCL allocated objects and host buffers.
  clStatus = clReleaseKernel(kernel);
  clStatus = clReleaseProgram(program);
  clStatus = clReleaseMemObject(x_clmem);
  clStatus = clReleaseMemObject(y_clmem);
  clStatus = clReleaseMemObject(z_clmem);
  clStatus = clReleaseCommandQueue(command_queue);
  clStatus = clReleaseContext(context);
  free(x);
  free(y);
  free(z);
  free(platforms);
  free(device_list);
  return 0;
}
