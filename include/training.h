#pragma once

#include <enum_util.h>
#include <move.h>
#include <nnue_model.h>
#include <search_constants.h>
#include <thread_worker.h>

#include <atomic>
#include <chrono>
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
  loss
};

constexpr char result_to_char(const result_type& result){
  switch(result){
    case result_type::win: return 'w';
    case result_type::draw: return 'd';
    case result_type::loss: return 'l';
    default: return 'd';
  }
}

constexpr result_type result_from_char(const char& result){
  switch(result){
    case 'w': return result_type::win;
    case 'd': return result_type::draw;
    case 'l': return result_type::loss;
    default: return result_type::draw;
  }
}

constexpr result_type mirrored_result(const result_type& result){
  switch(result){
    case result_type::win: return result_type::loss;
    case result_type::draw: return result_type::draw;
    case result_type::loss: return result_type::win;
    default: return result_type::draw;
  }
}

constexpr size_t half_feature_numel(){ return nnue::half_ka_numel; }
constexpr size_t max_active_half_features(){ return nnue::max_active_half_features; }

struct feature_set : chess::sided<feature_set, std::set<size_t>> {
  std::set<size_t> white;
  std::set<size_t> black;

  feature_set() : white{}, black{} {}
};

bool is_terminal(const chess::position_history& hist, const state_type& state) {
  if (hist.is_two_fold(state.hash())) { return true; }
  if (state.generate_moves().size() == 0) { return true; }

  return false;
}

result_type get_result(const chess::position_history& hist, const state_type& state) {
  if (hist.is_two_fold(state.hash())) { return result_type::draw; }
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
  state.show_init(features);
  return features;
}

}  // namespace train