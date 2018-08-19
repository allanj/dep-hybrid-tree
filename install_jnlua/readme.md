### JNLua Installation

- First install Torch in default directory ~/torch.
- Run `setup.sh` inside respective OS directory (currently Linux and MacOSX):
```
   cd Linux-Lua52 (or MacOSX-Lua52) && bash setup.sh
```
- For faster CPU computation, you may also need to put the following in `.bashrc`:
```
   export OMP_NUM_THREADS=16 (or your preferred #threads)
   export USE_OPENMP=1
```