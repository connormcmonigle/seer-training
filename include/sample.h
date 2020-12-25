#pragma once

#include <iostream>
#include <memory>
#include <fstream>
#include <iterator>
#include <sstream>
#include <cstdint>
#include <tuple>
#include <optional>

#include <training.h>

namespace train{

wdl_type mirror_wdl(const wdl_type& x){
  const auto [w, d, l] = x;
  return wdl_type(l, d, w);
}


struct sample{
  static constexpr char field_delimiter = '|';

  state_type state_{};

  score_type win_;
  score_type draw_;
  score_type loss_;

  feature_set features(){ return get_features(state_); }

  double win() const { return static_cast<double>(win_) / static_cast<double>(wdl_scale); }
  double draw() const { return static_cast<double>(draw_) / static_cast<double>(wdl_scale); }
  double loss() const { return static_cast<double>(loss_) / static_cast<double>(wdl_scale); }

  std::string to_string() const {
    return state_.fen() + sample::field_delimiter + 
    std::to_string(win_) + sample::field_delimiter +
    std::to_string(draw_) + sample::field_delimiter +
    std::to_string(loss_);
  }

  static sample from_string(const std::string& sample_str){
    sample x{};
    std::stringstream ss(sample_str);

    std::string sample_field{}; 
  
    std::getline(ss, sample_field, field_delimiter);
    x.state_ = state_type::parse_fen(sample_field);
    std::getline(ss, sample_field, field_delimiter);
    x.win_ = std::stoi(sample_field); 
    std::getline(ss, sample_field, field_delimiter);
    x.draw_ = std::stoi(sample_field); 
    std::getline(ss, sample_field, field_delimiter);
    x.loss_ = std::stoi(sample_field); 
    return x;
  }

  sample(const state_type& state, const wdl_type& wdl) : state_{state}, win_{std::get<0>(wdl)}, draw_{std::get<1>(wdl)}, loss_{std::get<2>(wdl)} {}
  sample(){}
};


template<typename T>
struct file_reader_iterator{
  using difference_type = long;
  using value_type = T;
  using pointer = const T*;
  using reference = const T&;
  using iterator_category = std::output_iterator_tag;

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

struct sample_reader{
  std::string path_;

  file_reader_iterator<sample> begin() const { return file_reader_iterator<sample>(to_line_reader<sample>(sample::from_string), path_); }

  file_reader_iterator<sample> end() const { return file_reader_iterator<sample>(); }

  size_t size() const {
    size_t size_{};
    for(const auto& _ : *this){ (void)_; ++size_; }
    return size_;
  }  

  sample_reader(const std::string& path) : path_{path} {}
};

struct sample_writer{
  std::string path_;
  std::ofstream file_;

  sample_writer& append_sample(const sample& datum){
    file_ << datum.to_string() << '\n';
    return *this;
  }

  sample_writer(const std::string& path) : path_{path}, file_{path} {}
};

}
