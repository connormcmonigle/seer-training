#pragma once

#include <optional>
#include <fstream>

namespace train{

template<typename T>
struct line_count_size{
  std::optional<size_t> size_{std::nullopt};

  std::string path() const { return static_cast<const T&>(*this).path_; }

  size_t compute_size_() const {
    std::ifstream file(path());
    size_t length{0};
    std::string line{};
    while(std::getline(file, line)){ ++length; }
    return length;
  }

  size_t size() const { return size_.value_or(compute_size_()); }
  
  size_t size(){
    if(size_.has_value()){ return size_.value(); }
    size_ = compute_size_();
    return size_.value();
  }

 private:
  line_count_size(){};
  friend T;
};

}