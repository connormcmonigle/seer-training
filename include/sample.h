#pragma once

#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <cstdint>
#include <tuple>

#include <training.h>

namespace train{

constexpr char field_delimiter = '|';

struct sample{
    state_type state_{};

    score_type win_;
    score_type draw_;
    score_type loss_;

    feature_set features(){ return get_features(state_); }

    double win() const { return static_cast<double>(win_) / static_cast<double>(wdl_scale); }
    double draw() const { return static_cast<double>(draw_) / static_cast<double>(wdl_scale); }
    double loss() const { return static_cast<double>(loss_) / static_cast<double>(wdl_scale); }

  sample(const state_type& state, const wdl_type& wdl) : state_{state}, win_{std::get<0>(wdl)}, draw_{std::get<1>(wdl)}, loss_{std::get<2>(wdl)} {}
  sample(){}
};

std::ostream& operator<<(std::ostream& ostr, const sample& x){
  return ostr << x.state_.fen() << field_delimiter << x.win_ << field_delimiter << x.draw_ << field_delimiter << x.loss_ << '\n';
}

std::istream& operator>>(std::istream& istr, sample& x){
  std::string sample_str{}; std::getline(istr, sample_str);
  if(!istr){ return istr; }

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
  return istr; 
}


struct sample_reader_iterator{
  using difference_type = long;
  using value_type = sample;
  using pointer = const sample*;
  using reference = const sample&;
  using iterator_category = std::output_iterator_tag;

  bool is_end_;
  std::ifstream file_;
  sample current_{};

  sample_reader_iterator& operator++(){
    is_end_ = !(file_ >> current_);
    return *this;
  }
  
  bool operator==(const sample_reader_iterator& other) const {
    return is_end_ && other.is_end_;
  }
  
  bool operator!=(const sample_reader_iterator& other) const {
    return !(*this == other);
  }

  sample operator*() const { return current_; }

  sample_reader_iterator(const std::string& path) : is_end_{false}, file_(path) {
    is_end_ = !(file_ >> current_);
  }

  sample_reader_iterator() : is_end_{true} {}
};

struct sample_reader{
  std::string path_;

  sample_reader_iterator begin() const { return sample_reader_iterator(path_); }
  sample_reader_iterator end() const { return sample_reader_iterator(); }

  sample_reader(const std::string& path) : path_{path} {}
};

struct sample_writer{
  std::string path_;
  std::ofstream file_;

  sample_writer& append_sample(const sample& datum){
    file_ << datum;
    return *this;
  }

  sample_writer(const std::string& path) : path_{path}, file_{path} {}
};

}
