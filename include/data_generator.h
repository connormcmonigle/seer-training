#pragma once

#include <cassert>
#include <iostream>
#include <optional>
#include <functional>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <filesystem>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <tuple>
#include <random>

#include <training.h>
#include <embedded_weights.h>

#include <sample.h>
#include <sample_reader.h>
#include <sample_writer.h>
#include <data_writer.h>


namespace train{

constexpr size_t half_feature_numel(){ return half_feature_numel_; }
constexpr size_t max_active_half_features(){ return max_active_half_features_; }

struct data_generator{
  size_t concurrency_{1};
  search::depth_type ply_limit_{128};
  search::depth_type random_ply_{10};
  search::depth_type fixed_depth_{6};


  std::shared_ptr<chess::transposition_table> tt_{nullptr};
  std::shared_ptr<search::constants> constants_ = std::make_shared<search::constants>(1);


  data_writer writer_;

  data_generator& set_concurrency(const size_t& concurrency){
    concurrency_ = concurrency;
    return *this;
  }

  data_generator& set_fixed_depth(const search::depth_type& fixed_depth){
    fixed_depth_ = fixed_depth;
    return *this;
  }

  data_generator& set_ply_limit(const search::depth_type& ply_limit){
    ply_limit_ = ply_limit;
    return *this;
  }

    data_generator& set_random_ply(const search::depth_type& random_ply){
    random_ply_ = random_ply;
    return *this;
  }

  data_generator& generate_data() {
    nnue::weights<real_type> weights{};
    nnue::embedded_weight_streamer<real_type> embedded(embed::weights_file_data);
    weights.load(embedded);
    
    auto generate = [&, this]{
      using worker_type = chess::thread_worker<real_type, false>;

      std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(&weights, tt_, constants_, [&, this](const auto& w) {
        if (w.depth() >= fixed_depth_) { worker->stop(); }
      });

      while (!writer_.is_complete()) {
        (worker->internal).reset();
        
        std::vector<sample> block{};

        chess::position_history hist{};
        state_type state = state_type::start_pos();
        
        for (search::depth_type ply = 0; ply < ply_limit_ && !std::get<0>(terminality(hist, state)); ++ply) {
          if (ply < random_ply_) {
            const auto mv_list = state.generate_moves();
            hist.push_(state.hash());
            state = state.forward(mv_list[std::rand() % mv_list.size()]);
          } else {
            worker->go(hist, state, 1);
            worker->iterative_deepening_loop_();
            
            const auto best_move = worker->best_move();

            if (std::abs(worker->score()) >= -search::max_mate_score) { break; }
            if (best_move.is_quiet() && !state.is_check()) block.emplace_back(state, worker->score());

            hist.push_(state.hash());
            state = state.forward(best_move);
          }
        }

        writer_.write_block(block);
      }
    };

    std::vector<std::thread> threads{};
    for (size_t i(0); i < concurrency_; ++i) { threads.emplace_back(generate); }
    for (auto& thread : threads) { thread.join(); }
    return *this;
  }

  data_generator(const std::string& path, const size_t& total, const size_t& tt_mb_size) : writer_{path, total} {
    tt_ = std::make_shared<chess::transposition_table>(tt_mb_size);
  }
};

}
