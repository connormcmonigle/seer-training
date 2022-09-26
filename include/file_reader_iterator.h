#pragma once

#include <fstream>
#include <memory>
#include <optional>
#include <functional>
#include <iterator>

namespace train{

template<typename T>
struct file_reader_iterator{
  using difference_type = long;
  using value_type = T;
  using pointer = const T*;
  using reference = const T&;
  using iterator_category = std::input_iterator_tag;

  std::optional<T> current_{std::nullopt};
  std::function<std::optional<T>(std::ifstream&)> read_element_;
  std::shared_ptr<std::ifstream> file_{nullptr};

  file_reader_iterator<T>& operator++(){
    current_ = read_element_(*file_);
    return *this;
  }

  file_reader_iterator<T> operator++(int){
    auto retval = *this;
    ++(*this);
    return retval;
  }

  bool operator==(const file_reader_iterator<T>& other) const {
    return !current_.has_value() && !other.current_.has_value();
  }
  
  bool operator!=(const file_reader_iterator<T>& other) const {
    return !(*this == other);
  }

  T operator*() const { return current_.value(); }

  template<typename F>
  file_reader_iterator(F&& f, const std::string& path) : read_element_{f}, file_{std::make_shared<std::ifstream>(path)} {
    ++(*this);
  }

  file_reader_iterator(){}
};

template<typename T, typename F>
auto to_line_reader(F&& f){
  return [f](std::ifstream& in){
    std::string representation{}; std::getline(in, representation);
    if(in){ return std::optional<T>(f(representation));}
    return std::optional<T>(std::nullopt);
  };
}

}