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

#include <embedded_weights.h>
#include <nnue_util.h>

#include <sample.h>
#include <sample_reader.h>
#include <sample_writer.h>
#include <data_writer.h>


namespace train{


struct data_generator{
  size_t concurrency_{1};

  search::depth_type ply_limit_{256};
  search::depth_type random_ply_{10};
  search::depth_type fixed_depth_{6};
  score_type eval_limit_{6144};

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

  data_generator& set_eval_limit(const score_type& eval_limit){
    eval_limit_ = std::abs(eval_limit);
    return *this;
  }

  data_generator& generate_data() {
    nnue::weights<real_type> weights{};
    nnue::embedded_weight_streamer<real_type> embedded(embed::weights_file_data);
    weights.load(embedded);
    
    auto generate = [&, this]{
      using worker_type = chess::thread_worker<real_type, false>;
      auto gen = std::mt19937(std::random_device()());

      std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(&weights, tt_, constants_, [&, this](const auto& w) {
        if (w.depth() >= fixed_depth_) { worker->stop(); }
      });

      while (!writer_.is_complete()) {
        (worker->internal).reset();
        
        std::vector<sample> block{};

        chess::position_history hist{};
        state_type state = state_type::start_pos();

        const result_type game_result = [&]{ 
          for (search::depth_type ply = 0; ply <= ply_limit_ && !is_terminal(hist, state); ++ply) {
            if (ply < random_ply_) {
              const auto mv_list = state.generate_moves();
              const size_t idx = std::uniform_int_distribution<size_t>(0, mv_list.size()-1)(gen);

              hist.push_(state.hash());
              state = state.forward(mv_list[idx]);
            } else {
              worker->go(hist, state, 1);
              worker->iterative_deepening_loop_();
              worker->stop();

              const auto best_move = worker->best_move();
              const auto best_score = worker->score();

              if (best_score >= eval_limit_) { return result_type::win; }
              if (best_score <= -eval_limit_) { return result_type::loss; }

              const auto view = search::stack_view::root((worker->internal).stack);
              const auto evaluator = [&] {
                nnue::eval<real_type> result(&weights);
                state.show_init(result);
                return result;
              }();
            
              const search::score_type static_eval = evaluator.evaluate(state.turn());

              worker->go(hist, state, 1);
              const search::score_type q_eval = worker->q_search<true, false>(view, evaluator, state, search::mate_score, -search::mate_score, 0);
              worker->stop();

              if (!state.is_check() && static_eval == q_eval) { block.emplace_back(state, best_score); }

              hist.push_(state.hash());
              state = state.forward(best_move);
            }
          }

          return get_result(hist, state);
        }();

        for (auto& elem : block) { elem.set_result(relative_result(state.turn(), elem.pov(), game_result)); }

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
