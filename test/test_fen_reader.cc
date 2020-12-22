#include <iostream>
#include <fstream>

#include <sample.h>
#include <session.h>



int main(){
  train::raw_fen_reader f_reader("test_fen.txt");
  for(const auto& elem : f_reader){ std::cout << elem.fen() << std::endl; }
}
