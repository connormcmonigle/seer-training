#include <iostream>

#include <session.h>

int main(){
  const std::string root_path = "/media/connor/7F35A067038168A9/seer_train/data_root";
  constexpr size_t man = 7;

  train::session sess{root_path};

  const auto train_path = sess.get_n_man_train_path(man);
  const auto raw_path = sess.get_n_man_raw_path(man);

  train::sample_reader sample_reader(train_path);
  const auto& const_sample_reader = sample_reader;

  std::cout << "const_sample_reader.size() :: " << const_sample_reader.size() << std::endl;
  std::cout << "sample_reader.size() :: " << sample_reader.size() << std::endl;
  std::cout << "const_sample_reader.size() :: " << const_sample_reader.size() << std::endl;
  std::cout << "sample_reader.size() :: " << sample_reader.size() << std::endl;

  train::raw_fen_reader raw_reader(raw_path);
  const auto& const_raw_reader = raw_reader;

  std::cout << "const_raw_reader.size() :: " << const_raw_reader.size() << std::endl;
  std::cout << "raw_reader.size() :: " << raw_reader.size() << std::endl;
  std::cout << "const_raw_reader.size() :: " << const_raw_reader.size() << std::endl;
  std::cout << "raw_reader.size() :: " << raw_reader.size() << std::endl;
}