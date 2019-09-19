## Ranger-Based iterative Random Forest
Ashley Cliff and Jonathon Romero

### Introduction
Iterative Random Forest (iRF) is an improvement upon the classic Random Forest, using weighted iterations to distill the forests. Ranger is a C++ implementation of random forest (Breiman 2001) or recursive partitioning, particularly suited for high dimensional data. Ranger-Based iRF uses Ranger as a base Random Forest codebase upon which iterative and scalability aspects have been added.

### Installation
To install in Linux or Mac OS X you will need a compiler supporting C++11 (i.e. gcc >= 4.7 or Clang >= 3.0) and Cmake. You will also, currently, need to have the Open-MPI libraries installed, even if you do not wish to use the MPI version. To build, start a terminal from the iRF main directory and run the following commands:

```bash
cd cpp_version
mkdir build
cd build
cmake ..
make
```

After compilation there should be an executable called "ranger" in the build directory. 

### Usage
#### Standalone C++ version
In the C++ version type 

```bash
ranger --help 
```

for a list of commands. First you need a training dataset in a file. This file should contain one header line with variable names and one line with variable values per sample. Variable names must not contain any whitespace, comma or semicolon. Values can be seperated by whitespace, comma or semicolon but can not be mixed in one file. The 'useMPI' flag must be used, followed by a 0 (NO) or a 1 (YES). Similarly, for the 'printPathfile' flag (however currently inverted, 0 is YES and 1 is NO). The 'numIterations' flag specifies the number of iterations; the flag and value are required and a typical default value to use would be 5. The 'outputDirectory' flag should be followed by the full path to the directory where any output will be written to, include log files and pathfiles (Do not include the trailing '/' at the end of the path). The 'outprefix' flag should be followed by the term that will be used as the name of each output file. A typical call would be for example

```bash
ranger  --file data.txt --depvarname Species --treetype 1 --ntree 1000 --nthreads 4 --useMPI 1 --numIterations 5 --outputDirectory /Users/user/Desktop -outprefix testRun --printPathfile 0
```

The ntree flag denotes the number of trees to create Per Compute Node.

It is not possible to do prediction with the useMPI flag at 1, and you will receive an error message if you try. As prediction does not require building trees, a single compute node (from a HPC system) is sufficient.

This is a beta code release, so there may be bugs and issues to work out. If you find any bugs or have any issues using the code, please talk to us and we will be glad to help. 

### References
* Basu, S., Kumbier, K., Brown, J. B. & Yu, B. Iterative random forests to discover predictive and stable high-order interactions. Proc. Natl. Acad. Sci. U. S. A. 115, 1943–1948 (2018).
* Wright, M. N. & Ziegler, A. (2017). ranger: A Fast Implementation of Random Forests for High Dimensional Data in C++ and R. Journal of Statistical Software 77:1-17. http://dx.doi.org/10.18637/jss.v077.i01.

