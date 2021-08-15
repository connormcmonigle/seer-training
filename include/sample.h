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
  result_type result_{result_type::undefined};

  sample& set_result(const result_type& result) {
    result_ = result;
    return *this;
  }

  bool pov() const { return state_.turn(); }

  feature_set features(){ return get_features(state_); }

  double score() const { return static_cast<double>(score_) / search::logit_scale<double>; }

  double result() const {
    switch(result_){
      case result_type::win: return 1.0;
      case result_type::draw: return 0.5;
      case result_type::loss: return 0.0;
      default: return sigmoid(score());
    }
  }

  std::string to_string() const {
    return state_.fen() + sample::field_delimiter + std::to_string(score_) + sample::field_delimiter + result_to_char(result_);
  }

  sample mirrored() const { return sample(state_.mirrored(), score_).set_result(result_); }

  static sample from_string(const std::string& sample_str){
    sample x{};
    std::stringstream ss(sample_str);

    std::string sample_field{}; 
  
    std::getline(ss, sample_field, field_delimiter);
    x.state_ = state_type::parse_fen(sample_field);
    std::getline(ss, sample_field, field_delimiter);
    x.score_ = std::stoi(sample_field); 
    if (std::getline(ss, sample_field, field_delimiter)) { x.result_ = result_from_char(sample_field[0]); }
    return x;
  }

  sample(const state_type& state, const score_type& score) : state_{state}, score_{score} {}
  sample(){}
};

}
