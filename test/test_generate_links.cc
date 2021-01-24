#include <iostream>

#include <session.h>

int main(){
  const std::string root_path = "/media/connor/7F35A067038168A9/seer_train/data_root";
  const std::string weights = "../scripts/model/save.bin";

  constexpr size_t man = 24;
  constexpr size_t concurrency = 12;

  train::session sess{root_path};
  sess.set_concurrency(concurrency);
  sess.load_weights(weights);

  sess.get_n_man_train_reader(man, [](const size_t& complete, const size_t& total){
    std::cout << complete << '/' << total << '\n';
  });
}