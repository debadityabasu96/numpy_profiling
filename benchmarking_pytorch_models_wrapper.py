import sys
import argparse
import torch
import timeit
import setuptools
import torch.utils
import torch.nn as nn
import torch.utils.benchmark
import torchvision.models as models
import os
def batched_dot_mul_sum(a,b):
    return a.mul(b).sum(-1)

def batched_dot_bmm(a,b):
    a = a.reshape(-1,1,a.shape[-1])
    b = b.reshape(-1,b.shape[-1],1)
    return torch.bmm(a,b).flatten(-3)

def run_model(model_name, input_vector):
    if (model_name == 'resnet18'):
        model = models.resnet18()
    elif (model_name == 'alexnet'):
        model = models.alexnet()
    elif(model_name=='vgg16'):
        model = models.vgg16()
    elif(model_name=='googlenet'):
        model = models.googlenet()
    elif(model_name=='shufflenet'):
        model = models.shufflenet_v2_x1_0()
    elif(model_name=='mobilenet_v2'):
        model = models.mobilenet_v2()
    elif (model_name == 'mobilenet_v3_large'):
        model = models.mobilenet_v3_large()
    elif(model_name=='mobilenet_v3_small'):
        model = models.mobilenet_v3_small()
    elif(model_name=='densenet'):
        model = models.densenet161()
    elif(model_name=='inception'):
        model = models.inception_v3()
    elif(model_name=='squeezenet'):
        model = models.squeezenet1_0()
    elif(model_name=='efficient_b0'):
        model = models.efficient_b0()
    elif(model_name=='efficient_b1'):
        model = models.efficient_b1()
    elif(model_name=='efficient_b2'):
        model = models.efficient_b2()
    elif(model_name=='efficient_b3'):
        model = models.efficient_b3()
    elif(model_name=='efficient_b4'):
        model = models.efficient_b4()
    elif(model_name=='efficient_b5'):
        model = models.efficient_b5()
    elif(model_name=='efficient_b6'):
        model = models.efficient_b6()
    elif(model_name=='efficient_b7'):
        model = models.efficient_b7()
    elif(model_name=='wide_resnet50_2'):
        model = models.wide_resnet50_2()
    else:
        model = models.convnext_tiny()
        print(type(model))
        model = getattr(models,model_name)
        print(type(model))
        return model(input_vector)

    input_tensor = torch.rand(input_vector)
    return model(input_tensor)


x = torch.randn(10000,64)

def tuple_type(strings):
    strings = strings.replace("(","").replace(")","")
    mapped_int = map(int,strings.split(","))
    return tuple(mapped_int)

def model_runtime(model,input_vector):
    t2 = torch.utils.benchmark.Timer(
            stmt='run_model(model,input_vector)',
            setup = 'from __main__ import run_model',
            globals={'model':model,
                'input_vector':input_vector})

    latency = t2.timeit(2000)
    return latency

def main():

    model = sys.argv[1]
    input_vector = sys.argv[2]
    input_vector = tuple(map(int,input_vector.split(',')))

    t0 = torch.utils.benchmark.Timer(
        stmt='batched_dot_mul_sum(x,x)',
        setup='from __main__ import batched_dot_mul_sum',
        globals={'x':x})

    t1 = torch.utils.benchmark.Timer(   
        stmt='batched_dot_bmm(x,x)',
        setup='from __main__ import batched_dot_bmm',
       globals={'x':x})

    
    latency = model_runtime(model,input_vector)
    

    print(t0.timeit(100))
    print(t1.timeit(100))
    
    #latency = (t2.timeit(100))
    print(latency)
    m0 = t0.blocked_autorange()
    m1 = t1.blocked_autorange()

    print(m0)
    print(m1)
if __name__ == "__main__":
    main()
