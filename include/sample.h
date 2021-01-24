#pragma once

#include <memory>
#include <fstream>
#include <string>
#include <sstream>

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

  bool pov() const { return state_.turn(); }
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

}
