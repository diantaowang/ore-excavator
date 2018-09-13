#include "../inc/aocl_utils.h"

// Print the error associciated with an error code
void printError(cl_int error)
{
	// Print error message
	switch (error) {
	case -1:
		printf("CL_DEVICE_NOT_FOUND ");
		break;
	case -2:
		printf("CL_DEVICE_NOT_AVAILABLE ");
		break;
	case -3:
		printf("CL_COMPILER_NOT_AVAILABLE ");
		break;
	case -4:
		printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
		break;
	case -5:
		printf("CL_OUT_OF_RESOURCES ");
		break;
	case -6:
		printf("CL_OUT_OF_HOST_MEMORY ");
		break;
	case -7:
		printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
		break;
	case -8:
		printf("CL_MEM_COPY_OVERLAP ");
		break;
	case -9:
		printf("CL_IMAGE_FORMAT_MISMATCH ");
		break;
	case -10:
		printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
		break;
	case -11:
		printf("CL_BUILD_PROGRAM_FAILURE ");
		break;
	case -12:
		printf("CL_MAP_FAILURE ");
		break;
	case -13:
		printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
		break;
	case -14:
		printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
		break;

	case -30:
		printf("CL_INVALID_VALUE ");
		break;
	case -31:
		printf("CL_INVALID_DEVICE_TYPE ");
		break;
	case -32:
		printf("CL_INVALID_PLATFORM ");
		break;
	case -33:
		printf("CL_INVALID_DEVICE ");
		break;
	case -34:
		printf("CL_INVALID_CONTEXT ");
		break;
	case -35:
		printf("CL_INVALID_QUEUE_PROPERTIES ");
		break;
	case -36:
		printf("CL_INVALID_COMMAND_QUEUE ");
		break;
	case -37:
		printf("CL_INVALID_HOST_PTR ");
		break;
	case -38:
		printf("CL_INVALID_MEM_OBJECT ");
		break;
	case -39:
		printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
		break;
	case -40:
		printf("CL_INVALID_IMAGE_SIZE ");
		break;
	case -41:
		printf("CL_INVALID_SAMPLER ");
		break;
	case -42:
		printf("CL_INVALID_BINARY ");
		break;
	case -43:
		printf("CL_INVALID_BUILD_OPTIONS ");
		break;
	case -44:
		printf("CL_INVALID_PROGRAM ");
		break;
	case -45:
		printf("CL_INVALID_PROGRAM_EXECUTABLE ");
		break;
	case -46:
		printf("CL_INVALID_KERNEL_NAME ");
		break;
	case -47:
		printf("CL_INVALID_KERNEL_DEFINITION ");
		break;
	case -48:
		printf("CL_INVALID_KERNEL ");
		break;
	case -49:
		printf("CL_INVALID_ARG_INDEX ");
		break;
	case -50:
		printf("CL_INVALID_ARG_VALUE ");
		break;
	case -51:
		printf("CL_INVALID_ARG_SIZE ");
		break;
	case -52:
		printf("CL_INVALID_KERNEL_ARGS ");
		break;
	case -53:
		printf("CL_INVALID_WORK_DIMENSION ");
		break;
	case -54:
		printf("CL_INVALID_WORK_GROUP_SIZE ");
		break;
	case -55:
		printf("CL_INVALID_WORK_ITEM_SIZE ");
		break;
	case -56:
		printf("CL_INVALID_GLOBAL_OFFSET ");
		break;
	case -57:
		printf("CL_INVALID_EVENT_WAIT_LIST ");
		break;
	case -58:
		printf("CL_INVALID_EVENT ");
		break;
	case -59:
		printf("CL_INVALID_OPERATION ");
		break;
	case -60:
		printf("CL_INVALID_GL_OBJECT ");
		break;
	case -61:
		printf("CL_INVALID_BUFFER_SIZE ");
		break;
	case -62:
		printf("CL_INVALID_MIP_LEVEL ");
		break;
	case -63:
		printf("CL_INVALID_GLOBAL_WORK_SIZE ");
		break;
	default:
		printf("UNRECOGNIZED ERROR CODE (%d)", error);
	}
    printf("\n");
}

void checkError(cl_int status)
{
	if (status != CL_SUCCESS)
		printError(status);
}

// Sets the current working directory to be the same as the directory
// containing the running executable.
bool setCwdToExeDir()
{
	// Get path of executable.
	char path[300];
	ssize_t n =
	    readlink("/proc/self/exe", path, sizeof(path) / sizeof(path[0]) - 1);
	if (n == -1) {
		return false;
	}
	path[n] = 0;

	// Find the last '\' or '/' and terminate the path there; it is now
	// the directory containing the executable.
	size_t i;
	for (i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i) ;
	path[i] = '\0';

	int rc;
	rc = chdir(path);

	return true;
}

// Loads a file in binary form.
unsigned char *loadBinaryFile(const char *file_name, size_t * size)
{
	// Open the File
	FILE *fp;
	fp = fopen(file_name, "rb");
	if (fp == 0) {
		return NULL;
	}
	// Get the size of the file
	fseek(fp, 0, SEEK_END);
	*size = ftell(fp);

	// Allocate space for the binary
	unsigned char *binary = (unsigned char *)malloc(*size);

	// Go back to the file start
	rewind(fp);

	// Read the file into the binary
	if (fread((void *)binary, *size, 1, fp) == 0) {
		free(binary);
		fclose(fp);
		return NULL;
	}
    fclose(fp);

	return binary;
}
