If you want to change CUDA versions in your current environment in python
Check current CUDA versions with nvidia-smi or nvcc –version
Check the current library path by print(os.environ["LD_LIBRARY_PATH"])
If the wrong version of CUDA is there update the path by these lines of code:

```
import os 
print(os.environ["PATH"]) #to check if correct CUDA version is in the path
LD_LIBRARY_PATH="/apps/icl/software/CUDAcompat/12.2-535.161.08/lib"
os.environ["LD_LIBRARY_PATH"]=os.pathsep+LD_LIBRARY_PATH
```
	
You should have the CUDA version:12.2 now 
You can check by nvidia-smi or nvcc –version
