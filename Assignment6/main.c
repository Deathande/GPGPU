#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>

#define MEM_SIZE 128
#define MAX_SOURCE_SIZE 0x100000

float random_float(float a, float b)
{
  return (float)rand() / ((float)RAND_MAX / (b-a)) + a;
}

float function(float x)
{
  return x * x;
}

float serial_mc(float a, float b, float height, long long throws, float(*f)(float))
{
  srand(time(NULL));
  int darts = 0;
  float rand_x;
  float rand_y;
  for (long long i = 0; i < throws; i++)
  {
    rand_x = random_float(a, b);
    rand_y = random_float(0, height);
    if (f(rand_x) < rand_y)
      darts++;
  }
  return darts / throws;
}


int main()
{
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobj = NULL;
  cl_program program = NULL;
  cl_kernel kernal = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;

  FILE* fp;
  char fname[] = "kernel.cl";
  char* source_str;
  size_t source_size;
  char string[MEM_SIZE];

  fp = fopen(fname, "r");
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  printf("\n\nGetting platform ID\'s\n");
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  if (ret != 0)
  {
    printf("Error getting platform id\n");
    return 1;
  }
  printf("Getting device ID\'s\n");
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
  if (ret != 0)
  {
    printf("Error getting device id\n");
    return 1;
  }
  printf("Creating context\n");
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  if (ret != 0)
  {
    printf("Error creating context\n");
    return 1;
  }
  printf("Creating command queue\n");
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if (ret != 0)
  {
    printf("Error creating command queue\n");
    return 1;
  }
  printf("Creating memory buffer\n");
  memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(char), NULL, &ret);
  if (ret != 0)
  {
    printf("Error creating memory buffer\n");
    return 1;
  }
  printf("Creating program with source\n");
  program = clCreateProgramWithSource(context,
                                      1,
                                      (const char **)&source_str,
                                      (const size_t *)&source_size,
                                      &ret);
  if (ret != 0)
  {
    printf("Error creating program from source\n");
    return 1;
  }
  printf("Compiling program\n");
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (ret != 0)
  {
    printf("Error compiling program\n");
    return 1;
  }
  
  printf("Creating kernal\n");
  kernal = clCreateKernel(program, "hello", &ret);
  if (ret != 0)
  {
    printf("Error creating kernel\n");
    return 1;
  }

  printf("Setting kernal args\n");
  ret = clSetKernelArg(kernal, 0, sizeof(cl_mem), (void*)&memobj);
  if (ret != 0)
  {
    printf("Error setting kernel args\n");
    return 1;
  }

  printf("Running kernel\n");
  ret = clEnqueueTask(command_queue, kernal, 0, NULL, NULL);
  if (ret != 0)
  {
    printf("Error running kernel\n");
    return 1;
  }
  printf("Copying results from device\n");
  ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(char), string, 0, NULL, NULL);
  if (ret != 0)
  {
    printf("Error copying from device\n");
    return 1;
  }

  puts(string);

  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernal);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);
  return 0;
}
