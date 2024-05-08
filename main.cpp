#include<numC/npArrayCpu.hpp>
#include<numC/npFunctions.hpp>
#include<numC/npRandom.hpp>
#include<numC/customKernels.hpp>

int main(){
auto A=np::ArrayCpu<float>(3,3,2);
auto B=np::ArrayCpu<float>(3,3,6);
std::cout<<A.dot(B);
return 0;
}