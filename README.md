# An Study on the Performance of Distributed Training of Data-driven CFD Simulations
Results presented in this section have been obtained with the following hardware:
- The predictive model training and inference is performed on the cluster CTE-Power. CTE-Power nodes are equipped each
with two processors IBM Power9 8335-GTH @ 2.4GHz with a total of 160 threads, 512GB of main memory, and four
GPU NVIDIA V100 with 16GB HBM2. Nodes are interconnected via single Port Mellanox EDR (25Gb/s).

Regarding the software stack, he predictive model has been developed on CTE-Power with Python 3.7.4, CUDA 10.2.89, Keras 2.4,
Tensorflow 2.3, Horovod 0.20.3, and OpenMPI 4.0.1.

Images with results are yet to be uploaded
