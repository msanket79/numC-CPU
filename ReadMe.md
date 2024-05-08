# numC: Accelerated CPU Library for C++, with NumPy syntax

numC is a C++ library inspired by numpy, designed for accelerated numerical computations on CPUs. It provides a familiar syntax similar to numpy, making it easy for users familiar with numpy to transition to CPU-accelerated computing seamlessly while working on C++.

Currently, numC only supports 2D arrays.

## Installation

To use numC, follow these steps:

1. Clone the repository:

   `git clone https://github.com/msanket79/numC.git`

2. Include whatever numC headers / functions you require:

   ```cpp
   #include "numC/npArrayCpu.hpp"
   ```

3. compile you numC including program using nvcc, and also compile and link gpuConfig.cu file.

## Features

### ArrayCPU Class

The ArrayCpu class provides functionalities for creating, manipulating, and performing operations on GPU-accelerated arrays. Here are some key features:

- Creation and initialization of arrays

```cpp
#include "numC/npArrayCpu.hpp"

int main(){
    auto A = np::ArrayCpu<float>(10, 2); // creates a 10x2 array filled with 0s
    A = np::ArrayCpu<float>(10, 2, 5); // 10x2 array with all elements as 5.

    int r = A.rows; // gets number of rows
    int c = A.cols; // gets number of cols
    int sz = A.size(); // gets size, rows * cols
}
```

- Reshaping arrays

```cpp
    A.reshape(5, 4); // reshapes it to 5x4 array.
    // reshape fails when new size does not match old size
```

- Getter, setter

```cpp
auto A = np::ArrayCpu<float>(10, 5);
float v = A.at(0); // assumes array as linear and gives number at position.
v = A.at(5, 4); // gives number at 5, 4 position.


/* numC also has support for indexing multiple elements at once, like numpy. */

auto C = A.at(np::ArrayCpu<int> idxs); // will return a 1D ArrayCpu with all the elements
                                       // from idxs given as parameter. order maintained.
// Ci = Ai where i is from idxs.

auto C = A.at(np::ArrayCpu<int> row, np::ArrayCpu<int> col); // Ci = A(rowi, coli).

// set function also has similar APIs
C.set(0, 5); // sets 0th index element as 5
C.set(np::ArrayCpu<int> idxs, np::ArrayCpu<float> val);
C.set(np::ArrayCpu<int> rows, np::ArrayCpu<int> cols, np::ArrayCpu<float> val);
```

- print function.

```cpp
auto A = np::ArrayCpu<float>(1024, 1024);
A.print(r,c); // prints whole array
std::cout<<A; // numC has overloaded << operator with cout, so cout also prints the full
              // array.
```

- Element-wise operations (addition, subtraction, multiplication, division). Returns a new array. (Supports broadcasting)

```cpp
    auto A = np::ArrayCpu<float>(10, 2);
    auto B = np::ArrayCpu<float>(10, 2);
    // currently only same type arrays are supported for operators.
    auto C = A + B;

    // broadcasting
    C = A + 5;
    C = A + np::ArrayCpu<float>(1, 2);
    C = A + np::ArrayCpu<float>(10, 1);
    C = A + np::ArrayCpu<float>(1, 1, 0);

    // shown for +, but also works with -, *, / operators
```

- Comparison operations (>, <, >=, <=, ==, !=). Returns array of 0s and 1s, depending on condition. (Supports broadcasting)

```cpp
    auto A = np::ArrayCpu<float>(10, 2);
    auto B = np::ArrayCpu<float>(10, 2);
    // currently only same type arrays are supported for operators.
    auto C = A < B;

    // broadcasting
    C = A < 5;
    C = A < np::ArrayCpu<float>(1, 2);
    C = A < np::ArrayCpu<float>(10, 1);
    C = A < np::ArrayCpu<float>(1, 1, 0);

    // shown for <, but also works with <=, >, >=, ==, != operators
```

- Transpose. returns a new transposed array.

```cpp
    auto A = np::ArrayCpu<float>(10, 2);
    auto AT = A.T();
```

- dot product - only supported for float32 dtype.

```cpp
auto A = np::ArrayCpu<float>(128, 1024);
auto B = np::ArrayCpu<float>(1024, 128);
auto C = A.dot(B);

// other dot functions
B = np::ArrayCpu<float>(128, 1024);
C = A.Tdot(B); // A transpose dot B

C = A.dotT(B); // A dot B transpose

```

- Statistical functions (sum, max, min, argmax, argmin). Returns a new array. additional argument - axis

```cpp
auto A = np::ArrayCpu<float>(128, 1024);

// default axis = -1. i.e. total sum
auto C = A.sum(); // returns 1x1 array
C = A.sum(0); // column wise sum. returns array of dimension 1x1024
C = A.sum(1); // row wise sum. returns array of dimension 128x1

/* works similarly with sum, max, min, argmax, argmin.
argmax, argmin return indexes of element instead of elements. return type is mandatorily
np::ArrayCpu<int> for these functions.*/
```

- Sort

```cpp
auto A = np::ArrayCpu<float>(128, 1024);

// default axis = -1. i.e.sort all elements considering A as 1d array
C=A.sort(); //returns sorted array
C = A.sort(0); //sorts the elements in every columnn
C=A.sort(1); // sorts the elements in each row

/* works similarly with sum, max, min, argmax, argmin.
argmax, argmin return indexes of element instead of elements. return type is mandatorily
np::ArrayCpu<int> for these functions.*/
```

### npFunctions header

- ones, zeros and arange

```cpp
#include "numC/npFunctions.hpp"


auto C = np::ones<float>(10); // 1d array of 1s
C = np::ones<float>(10, 10); // 2d array of 1s

C = np::zeros<float>(10, 10); // 1d array of 0s
C = np::zeros<float>(10, 10); // 2d array of 0s

C = np::arange<float>(10); // 1d array with numbers from 0 to 9, all at their respective
                           //  indexes. Immensely powerful for collective indexing, as
                           //  shown earlier
```

- maximum, minimum

```cpp
#include "numC/npFunctions.hpp"

auto A = np::ArrayCpu<float>(10, 5, 7); // 10x5 array, fill with 7
auto B = np::ArrayCpu<float>(10, 5, 6);

auto C = np::maximum(A, B);

// broadcasting
C = np::maximum(A, 0);
C = np::maximum(A, np::ArrayCpu<float>(10, 1));
C = np::maximum(A, np::ArrayCpu<float>(1, 5));

// works
```

- exp, log, square, sqrt, pow

```cpp
#include "numC/npFunctions.hpp"

auto A = np::ArrayCpu<float>(10, 5, 7); // 10x5 array, fill with 7

auto C = np::sqrt(A); // returns an array after applying function element wise.
// similar syntax for square, log, exp.

C = np::pow(A, 15); // returns an array after raising all elements by a power of pow.
                    // (pow is float)
```

- shuffle

```cpp
#include "numC/npFunctions.hpp"


auto A = np::arange<float>(1000); // array with numbers from 0 - 999

auto C = np::shuffle(A); // shuffles array randomly. (permutes)
```

- array_split

```cpp
#include "numC/npFunctions.hpp"


auto A = np::arange<float>(1000); // array with numbers from 0 - 999

auto batches = np::array_split(A, 5, 0); // array, num_parts, axis.
                                        // currently only axis = 0 is supported.
// returns a std::vector of arrays.
// will split even if arrays formed will be unequal.
// will create i%n arrays of size i/n + 1 and rest of size i/n
```

### Random Class

Random array generation (uniform and normal distributions)

- Uniform distribution

```cpp
#include "numC/npRandom.hpp"


auto A = np::Random::rand<float>(1, 100); // filled with numbers from uniform distribution
                                          // between 0 and 1
                                          // third argument can also be given - seed.
auto A = np::Random::rand<float>(1, 100, 20, 50); // filled with numbers from uniform
                                                  // distribution between 20 and 50
                                             // fifth argument can also be given - seed.
```

- Normal distribution

```cpp
#include "numC/npRandom.hpp"


auto A = np::Random::randn<float>(1, 100); // filled with numbers from normal distribution
                                           //  between 0 and 1
                                           // third argument can also be given - seed.
```

### Custom Kernels header

This has definitions of kernels of all functions we have used in numC which runs on CPU (except dot, dot is from openblas).

## Contribution and Future Development

While NumPy offers a vast array of commonly used functions such as  argsort and more, this project currently focuses on a specific set of functionalities. For our immediate needs, We've implemented the necessary functions; however, We may revisit this project in the future to expand its capabilities.

Contributions to enhance and add new functionalities are welcome! If you find areas where additional features could benefit the project or have ideas for improvements, feel free to contribute by opening an issue or submitting a pull request on GitHub.

## Acknowledgements

- openblas, used for dot product
- The kernels used here have been a result of lots of code browsing
- [duvanenko.tech.blog](https://duvanenko.tech.blog/) for parallel Sort

