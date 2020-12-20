#pragma once
#include <iomanip>
#include <iostream>
#include <cstdint>

#include <compat.h>

namespace train{

struct sample{
    compat::state_type state{};

    compat::score_type win_;
    compat::score_type draw_;
    compat::score_type loss_;

    double win() const { return static_cast<double>(win_) / static_cast<double>(compat::wdl_scale); }
    double draw() const { return static_cast<double>(draw_) / static_cast<double>(compat::wdl_scale); }
    double loss() const { return static_cast<double>(loss_) / static_cast<double>(compat::wdl_scale); }

};


std::ostream& operator<<(std::ostream& ostr, const sample& x){
  return ostr << x.state.fen() << '|' << x.win_ << '|' << x.draw_ << '|' << x.loss_ << '|';
}



}
