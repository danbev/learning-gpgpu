
saxpy: src/saxpy.c
	${CC} -g -o $@ $<

opencl-saxpy: src/opencl-saxpy.c
	${CC} -g -o $@ -lOpenCL $<
