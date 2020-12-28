#include <iostream>

#include <training.h>

int main(){
  train::train_interface<train::real_type> interface{};
  interface.load_weights("../scripts/model/save.bin");
  
  std::string fen{}; std::getline(std::cin, fen);
  const auto base = train::state_type::parse_fen(fen);
  
  const auto link = interface.get_continuation(base);

  if(link.has_value()){
    std::cout << link -> fen() << std::endl;
  }else{
    std::cout << "None" << std::endl;
  }
}