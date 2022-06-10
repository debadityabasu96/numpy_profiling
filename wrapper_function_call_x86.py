import os
import sys

def main():
    cpu_name = sys.argv[3]
    model = sys.argv[1]
    input_tensor = sys.argv[2]
    if(cpu_name != 'all'):
        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + cpu_name + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)
    else:
        
        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Haswell-v1' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)

        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Broadwell-v1' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)

        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Skylake-Server-v1' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)


        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Skylake-Client-v1' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)


        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Cooperlake-v1' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)


        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'Icelake-Client-v1' + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)


        bash_command = "qemu-x86_64-static" + " " + "-cpu" + " " + 'core2duo' + " " + "dist/benchmarking_pytorch_models_wrapper" + " " + model + " " + input_tensor
        os.system(bash_command)

if __name__ == "__main__":
    main()
