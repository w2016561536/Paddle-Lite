// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

class LwAxiRegs;

template <PrecisionType Ptype, PrecisionType OutType>
class CalibConv2dCompute : public KernelLite<TARGET(kIntelFPGA), Ptype> {
 public:
  void PrepareForRun() override;

  void ReInitWhenNeeded() override {
    // CHECK(impl_);
    // impl_->ReInitWhenNeeded();
  }

  void Run() override;

  ~CalibConv2dCompute() override;

 private:
  struct PackedStaticData {
    bool valid{false};
    int input_channels{0};
    int input_channels_aligned{0};
    int input_channel_blocks{0};
    int conv_out_channels{0};
    int depthwise_out_channels{0};
    int depthwise_kernel_h{0};
    int depthwise_kernel_w{0};
    float calib_scale{0.f};
    float depthwise_scale_value{0.f};

    size_t conv_weight_bytes{0};
    size_t depthwise_weight_bytes{0};
    size_t bias_conv_bytes{0};
    size_t bias_depthwise_bytes{0};
    size_t conv_scale_bytes{0};
    size_t depthwise_scale_bytes{0};

    std::vector<uint8_t> conv_weight;
    std::vector<uint8_t> depthwise_weight;
    std::vector<uint32_t> bias_conv;
    std::vector<uint32_t> bias_depthwise;
    std::vector<uint32_t> conv_scale;
    std::vector<uint32_t> depthwise_scale;

    void Clear() {
      valid = false;
      input_channels = 0;
      input_channels_aligned = 0;
      input_channel_blocks = 0;
      conv_out_channels = 0;
      depthwise_out_channels = 0;
      depthwise_kernel_h = 0;
      depthwise_kernel_w = 0;
      calib_scale = 0.f;
      depthwise_scale_value = 0.f;
      conv_weight_bytes = 0;
      depthwise_weight_bytes = 0;
      bias_conv_bytes = 0;
      bias_depthwise_bytes = 0;
      conv_scale_bytes = 0;
      depthwise_scale_bytes = 0;
      conv_weight.clear();
      depthwise_weight.clear();
      bias_conv.clear();
      bias_depthwise.clear();
      conv_scale.clear();
      depthwise_scale.clear();
    }
  };

  struct PersistentCmaData {
    bool valid{false};
    int fd{-1};
    void* map{nullptr};
    void* addr{nullptr};
    void* virt{nullptr};
    unsigned long phys{0};
    size_t size{0};
    size_t total_bytes{0};
    size_t used_bytes{0};

    int batch{0};
    int input_channels{0};
    int input_h{0};
    int input_w{0};
    int input_channels_aligned{0};
    int conv_out_channels{0};
    int output_channels{0};
    int output_h{0};
    int output_w{0};
    bool enable_conv2d_out{false};

    size_t input_bytes{0};
    size_t input_capacity_bytes{0};
    size_t conv_weight_bytes{0};
    size_t depthwise_weight_bytes{0};
    size_t bias_conv_bytes{0};
    size_t bias_depthwise_bytes{0};
    size_t conv_scale_bytes{0};
    size_t depthwise_scale_bytes{0};
    size_t output_fixed_bytes{0};
    size_t output_fixed_capacity_bytes{0};
    size_t conv2d_out_bytes{0};
    size_t conv2d_out_capacity_bytes{0};

    size_t off_input{0};
    size_t off_conv_w{0};
    size_t off_dw_w{0};
    size_t off_bias_conv{0};
    size_t off_output_fixed{0};
    size_t off_bias_dw{0};
    size_t off_output_conv2d{0};
    size_t off_conv_scale{0};
    size_t off_dw_scale{0};

    void ClearMeta() {
      valid = false;
      fd = -1;
      map = nullptr;
      addr = nullptr;
      virt = nullptr;
      phys = 0;
      size = 0;
      total_bytes = 0;
      used_bytes = 0;
      batch = 0;
      input_channels = 0;
      input_h = 0;
      input_w = 0;
      input_channels_aligned = 0;
      conv_out_channels = 0;
      output_channels = 0;
      output_h = 0;
      output_w = 0;
      enable_conv2d_out = false;
      input_bytes = 0;
      input_capacity_bytes = 0;
      conv_weight_bytes = 0;
      depthwise_weight_bytes = 0;
      bias_conv_bytes = 0;
      bias_depthwise_bytes = 0;
      conv_scale_bytes = 0;
      depthwise_scale_bytes = 0;
      output_fixed_bytes = 0;
      output_fixed_capacity_bytes = 0;
      conv2d_out_bytes = 0;
      conv2d_out_capacity_bytes = 0;
      off_input = 0;
      off_conv_w = 0;
      off_dw_w = 0;
      off_bias_conv = 0;
      off_output_fixed = 0;
      off_bias_dw = 0;
      off_output_conv2d = 0;
      off_conv_scale = 0;
      off_dw_scale = 0;
    }
  };

  struct HlsRegConfig {
    bool valid{false};
    unsigned long input_phys{0};
    unsigned long weight_conv_phys{0};
    unsigned long weight_dw_phys{0};
    unsigned long bias_conv_phys{0};
    unsigned long output_fixed_phys{0};
    unsigned long bias_dw_phys{0};
    unsigned long output_conv2d_phys{0};
    unsigned long conv_scale_phys{0};
    unsigned long dw_scale_phys{0};
    uint32_t inv_dw_scale{0};
    uint32_t input_hw{0};
    uint32_t input_channels{0};
    uint32_t out_channels{0};
    uint32_t dw_kernel{0};
    uint32_t dw_stride{0};
    uint32_t dw_padding{0};
    uint32_t enable_conv_out{0};
    uint32_t local_groups_per_round{0};
    uint32_t rounds{0};

    void Clear() {
      valid = false;
      input_phys = 0;
      weight_conv_phys = 0;
      weight_dw_phys = 0;
      bias_conv_phys = 0;
      output_fixed_phys = 0;
      bias_dw_phys = 0;
      output_conv2d_phys = 0;
      conv_scale_phys = 0;
      dw_scale_phys = 0;
      inv_dw_scale = 0;
      input_hw = 0;
      input_channels = 0;
      out_channels = 0;
      dw_kernel = 0;
      dw_stride = 0;
      dw_padding = 0;
      enable_conv_out = 0;
      local_groups_per_round = 0;
      rounds = 0;
    }

    bool SameAs(const HlsRegConfig& other) const {
      return valid && other.valid &&
             input_phys == other.input_phys &&
             weight_conv_phys == other.weight_conv_phys &&
             weight_dw_phys == other.weight_dw_phys &&
             bias_conv_phys == other.bias_conv_phys &&
             output_fixed_phys == other.output_fixed_phys &&
             bias_dw_phys == other.bias_dw_phys &&
             output_conv2d_phys == other.output_conv2d_phys &&
             conv_scale_phys == other.conv_scale_phys &&
             dw_scale_phys == other.dw_scale_phys &&
             inv_dw_scale == other.inv_dw_scale &&
             input_hw == other.input_hw &&
             input_channels == other.input_channels &&
             out_channels == other.out_channels &&
             dw_kernel == other.dw_kernel &&
             dw_stride == other.dw_stride &&
             dw_padding == other.dw_padding &&
             enable_conv_out == other.enable_conv_out &&
             local_groups_per_round == other.local_groups_per_round &&
             rounds == other.rounds;
    }
  };
  void ReleasePersistentCma();

  using param_t = operators::ConvParam;
  PackedStaticData packed_static_;
  PersistentCmaData cma_runtime_;
  HlsRegConfig reg_config_;
  std::vector<uint8_t> input_pack_cache_;
  std::vector<uint32_t> output_fixed_cache_;
  std::vector<uint32_t> conv2d_out_cache_;
  size_t input_pack_cache_bytes_{0};
  size_t output_fixed_cache_words_{0};
  size_t conv2d_out_cache_words_{0};
  LwAxiRegs* regs_{nullptr};
  std::unique_ptr<KernelContext> arm_cxt_{nullptr};
  KernelLite<TARGET(kARM), Ptype>* impl_{nullptr};
};

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
