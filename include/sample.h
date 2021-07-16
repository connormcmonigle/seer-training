#pragma once

#include <memory>
#include <fstream>
#include <string>
#include <sstream>

#include <training.h>

namespace train{

struct sample{
  static constexpr char field_delimiter = '|';

  state_type state_{};

  score_type score_;

  bool pov() const { return state_.turn(); }
  feature_set features(){ return get_features(state_); }

  double score() const { return static_cast<double>(score_) / search::logit_scale<double>; }

  std::string to_string() const {
    return state_.fen() + sample::field_delimiter + std::to_string(score_);
  }

  sample mirrored() const { return sample(state_.mirrored(), score_); }

  static sample from_string(const std::string& sample_str){
    sample x{};
    std::stringstream ss(sample_str);

    std::string sample_field{}; 
  
    std::getline(ss, sample_field, field_delimiter);
    x.state_ = state_type::parse_fen(sample_field);
    std::getline(ss, sample_field, field_delimiter);
    x.score_ = std::stoi(sample_field); 
    return x;
  }

  sample(const state_type& state, const score_type& score) : state_{state}, score_{score} {}
  sample(){}
};

}
