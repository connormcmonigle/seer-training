#include <iostream>
#include <fstream>

#include <sample.h>
#include <session.h>



int main(){
  train::sample_reader s_reader("test_sample.txt");
  for(const auto& elem : s_reader){ std::cout << elem.to_string() << std::endl; }
}