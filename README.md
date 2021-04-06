# An enhancement of fuzzy k-nearest neighbor classifier using multi-local power means
### Introduction: 
This is a new method to the family of fuzzy k-nearest neighbor (FKNN) classifiers based on the use of power means in the calculation of multi-local means that are used in classification of samples. This method is called multi-local power means fuzzy k-nearest neighbor classifier (MLPM-FKNN). The MLPM-FKNN can be adapted to the context (of different data sets), due to the power mean being parametric and thus allowing for testing to find the parameter value that can be optimized for the classification accuracy. 

### Matlab functions:

The function of the mlpm-fknn algorithm (`mlpm_fknn.m`) and Power mean computation (`pmean.m`) are included. In addition to those files, an example (`example_mlpm_fknn.m`) of the use of mlpm-fknn classifier is also presented. `pmean.m` is needed to compute Power mean vectors of the set of nearest neighbor in each class.

Reference: [Kumbure, M. M., Luukka, P., Collan, M.: An enhancement of fuzzy k-nearest neighbor classifier using multi-local power means. In: Proceeding of the 11th Conference of the European Society for Fuzzy Logic and Technology (EUSFLAT), pp. 83–	90, Atlantis Press (2019)](https://doi.org/10.2991/eusflat-19.2019.13)

Created by Mahinda Mailagaha Kumbure & Pasi Luukka 01/2019 based on Keller's definition of the fuzzy k-nearest neighbor algorithm.
