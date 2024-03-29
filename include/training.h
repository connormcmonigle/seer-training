#pragma once

#include <chess/types.h>
#include <feature/util.h>
#include <chess/move.h>
#include <chess/board_history.h>
#include <nnue/eval.h>
#include <search/search_constants.h>
#include <search/search_worker.h>
#include <nnue/embedded_weights.h>
#include <nnue/weights_streamer.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

namespace train {

using real_type = float;
using state_type = chess::board;
using score_type = search::score_type;

enum class result_type {
  win,
  draw,
  loss,
  undefined
};

constexpr char result_to_char(const result_type& result){
  switch(result){
    case result_type::win: return 'w';
    case result_type::draw: return 'd';
    case result_type::loss: return 'l';
    default: return 'u';
  }
}

constexpr result_type result_from_char(const char& result){
  switch(result){
    case 'w': return result_type::win;
    case 'd': return result_type::draw;
    case 'l': return result_type::loss;
    default: return result_type::undefined;
  }
}

constexpr result_type mirrored_result(const result_type& result){
  switch(result){
    case result_type::win: return result_type::loss;
    case result_type::draw: return result_type::draw;
    case result_type::loss: return result_type::win;
    default: return result_type::undefined;
  }
}

constexpr size_t half_feature_numel(){ return feature::half_ka::numel; }
constexpr size_t max_active_half_features(){ return feature::half_ka::max_active_half_features; }

real_type sigmoid(const real_type& x) {
  constexpr real_type one = static_cast<real_type>(1);
  return one / (std::exp(-x) + one);
}

struct feature_set : chess::sided<feature_set, std::set<size_t>> {
  std::set<size_t> white;
  std::set<size_t> black;

  feature_set() : white{}, black{} {}
};

bool is_terminal(const chess::board_history& hist, const state_type& state) {
  if (hist.count(state.hash())) { return true; }
  if (state.generate_moves().size() == 0) { return true; }

  return false;
}

result_type get_result(const chess::board_history& hist, const state_type& state) {
  if (hist.count(state.hash())) { return result_type::draw; }
  if (state.generate_moves().size() == 0) {
    if (state.is_check()) { return result_type::loss; }
    return result_type::draw;
  }

  return result_type::draw;
}

result_type relative_result(const bool& pov_a, const bool& pov_b, const result_type& result){
  return pov_a == pov_b ? result : mirrored_result(result);
}

feature_set get_features(const state_type& state) {
  feature_set features{};
  state.feature_full_reset(features);
  return features;
}

std::tuple<std::vector<nnue::weights::parameter_type>, std::vector<nnue::weights::parameter_type>> feature_transformer_parameters() {
  nnue::embedded_weight_streamer streamer(nnue::embed::weights_file_data);
  using feature_transformer_type = nnue::sparse_affine_layer<nnue::weights::parameter_type, feature::half_ka::numel, nnue::weights::base_dim>;
  feature_transformer_type feature_transformer = nnue::weights{}.load(streamer).shared;
  std::vector<nnue::weights::parameter_type> weights(feature_transformer.W, feature_transformer.W + feature_transformer_type::W_numel);
  std::vector<nnue::weights::parameter_type> bias(feature_transformer.b, feature_transformer.b + feature_transformer_type::b_numel);
  return std::tuple(weights, bias);
}

}  // namespace train