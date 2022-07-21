#include <iostream>
#include <fstream>

#include <sample.h>
#include <sample_reader.h>

int main(){
  train::sample_reader s_reader("test.txt");
  for(const auto& elem : s_reader){ std::cout << elem.to_string() << std::endl; }
}