#ifndef __AOCL_UTILS_
#define __AOCL_UTILS_

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include "CL/opencl.h"

void printError(cl_int error);
void checkError(cl_int status);
bool setCwdToExeDir(void);
unsigned char *loadBinaryFile(const char *file_name, size_t *size);

#endif
