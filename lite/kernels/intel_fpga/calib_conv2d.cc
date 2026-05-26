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

#include "lite/kernels/intel_fpga/calib_conv2d.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/intel_fpga/conv_depthwise.h"
#include "lite/kernels/intel_fpga/conv_gemmlike.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {
#define PARAM_INIT                                                           \
  auto& param = this->Param<param_t>();                                      \
  auto w_dims = param.filter->dims();                                        \
  auto paddings = *param.paddings;                                           \
  auto dilations = *param.dilations;                                         \
  int ic = w_dims[1] * param.groups;                                         \
  int oc = w_dims[0];                                                        \
  int kh = w_dims[2];                                                        \
  int kw = w_dims[3];                                                        \
  int pad_h = paddings[0];                                                   \
  int pad_w = paddings[2];                                                   \
  int stride = param.strides[0];                                             \
  int sh = param.strides[1];                                                 \
  int sw = param.strides[0];                                                 \
  int chin = param.x->dims()[1];                                             \
  int hin = param.x->dims()[2];                                              \
  int win = param.x->dims()[3];                                              \
  int chout = param.output->dims()[1];                                       \
  int hout = param.output->dims()[2];                                        \
  int wout = param.output->dims()[3];                                        \


template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  // PARAM_INIT
  /// select conv impl
  // if (param.groups == ic && ic == oc && ks_equal && no_dilation && flag_dw) {
  //   impl_ = new DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>;
  //   VLOG(3) << "[IntelFPGA] invoking depthwise conv";
  // } else {
  //   impl_ = new GemmLikeConv<PRECISION(kFloat), PRECISION(kFloat)>;
  //   VLOG(3) << "[IntelFPGA] invoking common conv";
  // }
  // if (!arm_cxt_) {
  //   arm_cxt_ = ContextScheduler::Global().NewContext(TargetType::kARM);
  // }
  std::cout << "FPGA calib_conv2d param init finish" << std::endl;

  // impl_->SetContext(std::move(arm_cxt_));
  // impl_->SetParam(param);
  // impl_->PrepareForRun();

}

template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();

  // ------------------------------------------------------------
  // Tensor data
  // ------------------------------------------------------------

  const float* x_data = param.x->data<float>();

  // 注意：这里的 filter tensor 虽然以 float 形式保存，
  // 但其数值语义是否已经是量化后的 int8 权重值，需要通过打印确认。
  const int8_t* conv_w_q_float = param.filter->data<int8_t>();
  const int8_t* dw_w_q_float = param.depthwise_filter->data<int8_t>();

  const float* conv_bias =
      param.bias ? param.bias->data<float>() : nullptr;

  const float* dw_bias =
      param.depthwise_bias ? param.depthwise_bias->data<float>() : nullptr;

  float* out_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto conv_w_dims = param.filter->dims();
  auto dw_w_dims = param.depthwise_filter->dims();
  auto out_dims = param.output->dims();

  const int batch = x_dims[0];
  const int in_c = x_dims[1];
  const int in_h = x_dims[2];
  const int in_w = x_dims[3];

  // ------------------------------------------------------------
  // 第一层 conv2d:
  // fixed 1x1 / stride=1 / padding=0 / dilation=1 / groups=1
  // filter layout: [OC, IC, 1, 1]
  // ------------------------------------------------------------

  const int conv_oc = conv_w_dims[0];
  const int conv_ic = conv_w_dims[1];
  const int conv_kh = conv_w_dims[2];
  const int conv_kw = conv_w_dims[3];

  CHECK_EQ(conv_kh, 1);
  CHECK_EQ(conv_kw, 1);
  CHECK_EQ(conv_ic, in_c);

  const int conv_out_h = in_h;
  const int conv_out_w = in_w;
  const int conv_out_numel = batch * conv_oc * conv_out_h * conv_out_w;

  // ------------------------------------------------------------
  // 第二层 depthwise_conv2d:
  // 使用 param.strides / param.paddings / param.dilations / param.groups
  // filter layout: [DW_OC, IC_PER_GROUP, KH, KW]
  // 对 depthwise 来说通常是 [C, 1, KH, KW]
  // ------------------------------------------------------------

  const int dw_oc = dw_w_dims[0];
  const int dw_ic_per_group = dw_w_dims[1];
  const int dw_kh = dw_w_dims[2];
  const int dw_kw = dw_w_dims[3];

  const int out_n = out_dims[0];
  const int out_c = out_dims[1];
  const int out_h = out_dims[2];
  const int out_w = out_dims[3];

  CHECK_EQ(out_n, batch);
  CHECK_EQ(out_c, dw_oc);

  // depthwise: groups 应等于输入通道数，也就是 conv2d 输出通道数
  CHECK_EQ(param.groups, conv_oc);
  CHECK_EQ(dw_ic_per_group, conv_oc / param.groups);
  CHECK_EQ(dw_ic_per_group, 1);
  CHECK_EQ(dw_oc % param.groups, 0);

  const int dw_channel_multiplier = dw_oc / param.groups;

  CHECK_EQ(param.Conv2d_Filter0_scale.size(), conv_oc);
  CHECK_EQ(param.Depthwise2d_Filter0_scale.size(), dw_oc);

  const float calib_scale = param.calib_scale;
  const float depthwise_scale = param.depthwise_scale;

  const auto& conv_filter_scale = param.Conv2d_Filter0_scale;
  const auto& depthwise_filter_scale = param.Depthwise2d_Filter0_scale;

  // ============================================================
  // Debug print: 在正式计算前打印所有权重、bias、scale、shape
  // ============================================================

  // std::cout << std::fixed ;

  // std::cout << "\n================ CalibConv2dCompute Debug Print ================\n";

  // std::cout << "[Input Tensor]\n";
  // std::cout << "  x_dims = ["
  //           << batch << ", "
  //           << in_c << ", "
  //           << in_h << ", "
  //           << in_w << "]\n";
  // std::cout << "  calib_scale = " << calib_scale << "\n";

  // std::cout << "\n[Conv2D Tensor]\n";
  // std::cout << "  conv_filter_dims = ["
  //           << conv_oc << ", "
  //           << conv_ic << ", "
  //           << conv_kh << ", "
  //           << conv_kw << "]\n";
  // std::cout << "  conv_out_dims = ["
  //           << batch << ", "
  //           << conv_oc << ", "
  //           << conv_out_h << ", "
  //           << conv_out_w << "]\n";

  // std::cout << "\n[Depthwise Conv2D Tensor]\n";
  // std::cout << "  depthwise_filter_dims = ["
  //           << dw_oc << ", "
  //           << dw_ic_per_group << ", "
  //           << dw_kh << ", "
  //           << dw_kw << "]\n";
  // std::cout << "  output_dims = ["
  //           << out_n << ", "
  //           << out_c << ", "
  //           << out_h << ", "
  //           << out_w << "]\n";
  // std::cout << "  groups = " << param.groups << "\n";
  // std::cout << "  dw_channel_multiplier = " << dw_channel_multiplier << "\n";
  // std::cout << "  depthwise_scale = " << depthwise_scale << "\n";

  // int stride_h_dbg = 1;
  // int stride_w_dbg = 1;
  // if (param.strides.size() >= 2) {
  //   stride_h_dbg = param.strides[0];
  //   stride_w_dbg = param.strides[1];
  // }

  // int dilation_h_dbg = 1;
  // int dilation_w_dbg = 1;
  // if (param.dilations && param.dilations->size() >= 2) {
  //   dilation_h_dbg = (*param.dilations)[0];
  //   dilation_w_dbg = (*param.dilations)[1];
  // }

  // int pad_top_dbg = 0;
  // int pad_bottom_dbg = 0;
  // int pad_left_dbg = 0;
  // int pad_right_dbg = 0;

  // if (param.paddings) {
  //   if (param.paddings->size() == 2) {
  //     pad_top_dbg = (*param.paddings)[0];
  //     pad_bottom_dbg = (*param.paddings)[0];
  //     pad_left_dbg = (*param.paddings)[1];
  //     pad_right_dbg = (*param.paddings)[1];
  //   } else if (param.paddings->size() == 4) {
  //     pad_top_dbg = (*param.paddings)[0];
  //     pad_bottom_dbg = (*param.paddings)[1];
  //     pad_left_dbg = (*param.paddings)[2];
  //     pad_right_dbg = (*param.paddings)[3];
  //   }
  // }

  // std::cout << "  strides = [" << stride_h_dbg << ", " << stride_w_dbg << "]\n";
  // std::cout << "  dilations = [" << dilation_h_dbg << ", " << dilation_w_dbg << "]\n";
  // std::cout << "  paddings = [top=" << pad_top_dbg
  //           << ", bottom=" << pad_bottom_dbg
  //           << ", left=" << pad_left_dbg
  //           << ", right=" << pad_right_dbg << "]\n";

  // // ------------------------------------------------------------
  // // Conv2D 权重、bias、scale 打印
  // // ------------------------------------------------------------

  const int conv_w_numel = conv_oc * conv_ic * conv_kh * conv_kw;

  // int conv_zero_after_cast = 0;
  // int conv_non_integer_count = 0;
  // float conv_min_w = 1e30f;
  // float conv_max_w = -1e30f;

  // for (int i = 0; i < conv_w_numel; ++i) {
  //   const float w = static_cast<float>(conv_w_q_float[i]);
  //   conv_min_w = std::min(conv_min_w, w);
  //   conv_max_w = std::max(conv_max_w, w);

  //   if (static_cast<int32_t>(w) == 0) {
  //     conv_zero_after_cast++;
  //   }

  //   if (std::fabs(w - std::round(w)) > 1e-3f) {
  //     conv_non_integer_count++;
  //   }
  // }

  // std::cout << "\n[Conv2D Weight Statistics]\n";
  // std::cout << "  conv_w_numel = " << conv_w_numel << "\n";
  // std::cout << "  conv_w_float_min = " << conv_min_w << "\n";
  // std::cout << "  conv_w_float_max = " << conv_max_w << "\n";
  // std::cout << "  conv_zero_after_static_cast_int32 = "
  //           << conv_zero_after_cast << " / " << conv_w_numel << "\n";
  // std::cout << "  conv_non_integer_like_weight = "
  //           << conv_non_integer_count << " / " << conv_w_numel << "\n";

  // std::cout << "\n[Conv2D Bias and Scale]\n";
  // for (int oc = 0; oc < conv_oc; ++oc) {
  //   std::cout << "  conv_oc=" << oc
  //             << ", filter_scale=" << conv_filter_scale[oc];

  //   if (conv_bias) {
  //     std::cout << ", bias=" << conv_bias[oc];
  //   } else {
  //     std::cout << ", bias=null";
  //   }

  //   std::cout << ", dequant_out_scale=calib_scale*filter_scale="
  //             << calib_scale * conv_filter_scale[oc] << "\n";
  // }

  // std::cout << "\n[Conv2D All Weights]\n";
  // for (int oc = 0; oc < conv_oc; ++oc) {
  //   std::cout << "  ---- conv_oc=" << oc << " ----\n";
  //   std::cout << "  scale=" << conv_filter_scale[oc];
  //   if (conv_bias) {
  //     std::cout << ", bias=" << conv_bias[oc];
  //   } else {
  //     std::cout << ", bias=null";
  //   }
  //   std::cout << "\n";

  //   for (int ic = 0; ic < conv_ic; ++ic) {
  //     for (int kh = 0; kh < conv_kh; ++kh) {
  //       for (int kw = 0; kw < conv_kw; ++kw) {
  //         const int idx =
  //             ((oc * conv_ic + ic) * conv_kh + kh) * conv_kw + kw;

  //         const float w_float = static_cast<float>(conv_w_q_float[idx]);
  //         const int32_t w_cast = static_cast<int32_t>(w_float);
  //         const int32_t w_round = static_cast<int32_t>(std::round(w_float));

  //         std::cout << "    conv_w"
  //                   << "[oc=" << oc
  //                   << "][ic=" << ic
  //                   << "][kh=" << kh
  //                   << "][kw=" << kw
  //                   << "]"
  //                   << " raw_float=" << w_float
  //                   << ", static_cast_int32=" << w_cast
  //                   << ", round_int32=" << w_round
  //                   << "\n";
  //       }
  //     }
  //   }
  // }

  // // ------------------------------------------------------------
  // // Depthwise 权重、bias、scale 打印
  // // ------------------------------------------------------------

  const int dw_w_numel = dw_oc * dw_ic_per_group * dw_kh * dw_kw;

  // int dw_zero_after_cast = 0;
  // int dw_non_integer_count = 0;
  // float dw_min_w = 1e30f;
  // float dw_max_w = -1e30f;

  // for (int i = 0; i < dw_w_numel; ++i) {
  //   const float w = static_cast<float>(dw_w_q_float[i]);
  //   dw_min_w = std::min(dw_min_w, w);
  //   dw_max_w = std::max(dw_max_w, w);

  //   if (static_cast<int32_t>(w) == 0) {
  //     dw_zero_after_cast++;
  //   }

  //   if (std::fabs(w - std::round(w)) > 1e-3f) {
  //     dw_non_integer_count++;
  //   }
  // }

  // std::cout << "\n[Depthwise Weight Statistics]\n";
  // std::cout << "  dw_w_numel = " << dw_w_numel << "\n";
  // std::cout << "  dw_w_float_min = " << dw_min_w << "\n";
  // std::cout << "  dw_w_float_max = " << dw_max_w << "\n";
  // std::cout << "  dw_zero_after_static_cast_int32 = "
  //           << dw_zero_after_cast << " / " << dw_w_numel << "\n";
  // std::cout << "  dw_non_integer_like_weight = "
  //           << dw_non_integer_count << " / " << dw_w_numel << "\n";

  // std::cout << "\n[Depthwise Bias and Scale]\n";
  // for (int oc = 0; oc < dw_oc; ++oc) {
  //   std::cout << "  dw_oc=" << oc
  //             << ", filter_scale=" << depthwise_filter_scale[oc];

  //   if (dw_bias) {
  //     std::cout << ", bias=" << dw_bias[oc];
  //   } else {
  //     std::cout << ", bias=null";
  //   }

  //   std::cout << ", dequant_out_scale=depthwise_scale*filter_scale="
  //             << depthwise_scale * depthwise_filter_scale[oc] << "\n";
  // }

  // std::cout << "\n[Depthwise All Weights]\n";
  // for (int oc = 0; oc < dw_oc; ++oc) {
  //   std::cout << "  ---- dw_oc=" << oc << " ----\n";
  //   std::cout << "  scale=" << depthwise_filter_scale[oc];
  //   if (dw_bias) {
  //     std::cout << ", bias=" << dw_bias[oc];
  //   } else {
  //     std::cout << ", bias=null";
  //   }
  //   std::cout << "\n";

  //   for (int icg = 0; icg < dw_ic_per_group; ++icg) {
  //     for (int kh = 0; kh < dw_kh; ++kh) {
  //       for (int kw = 0; kw < dw_kw; ++kw) {
  //         const int idx =
  //             ((oc * dw_ic_per_group + icg) * dw_kh + kh) * dw_kw + kw;

  //         const float w_float = static_cast<float>(dw_w_q_float[idx]);
  //         const int32_t w_cast = static_cast<int32_t>(w_float);
  //         const int32_t w_round = static_cast<int32_t>(std::round(w_float));

  //         std::cout << "    dw_w"
  //                   << "[oc=" << oc
  //                   << "][ic_per_group=" << icg
  //                   << "][kh=" << kh
  //                   << "][kw=" << kw
  //                   << "]"
  //                   << " raw_float=" << w_float
  //                   << ", static_cast_int32=" << w_cast
  //                   << ", round_int32=" << w_round
  //                   << "\n";
  //       }
  //     }
  //   }
  // }

  // std::cout << "================ End Debug Print ================\n\n";

  // ------------------------------------------------------------
  // helper functions
  // ------------------------------------------------------------

  auto hard_swish = [](float x) -> float {
    float t = x + 3.f;
    if (t < 0.f) {
      t = 0.f;
    } else if (t > 6.f) {
      t = 6.f;
    }
    return x * t / 6.f;
  };

  auto quantize_input_to_int8 = [](float x, float scale) -> int32_t {
    // 对齐 Paddle Lite calib:
    // q = roundf(x * (1.f / scale))
    // clip 到 [-127, 127]
    int32_t q = static_cast<int32_t>(roundf(x * (1.f / scale)));

    if (q > 127) {
      q = 127;
    } else if (q < -127) {
      q = -127;
    }

    return q;
  };

  auto load_quantized_weight = [](float w) -> int32_t {
    // 如果权重已经是量化后的整数值，只做类型转换。
    // 注意：这里为了严格保持你当前代码的行为，仍然使用 static_cast<int32_t>。
    // 如果 debug 打印显示权重是原始 float 小数，这里就是导致大量 0 的位置。
    return static_cast<int32_t>(w);
  };

  // ============================================================
  // 1. input float -> int8/int32
  // ============================================================

  const int input_numel = batch * in_c * in_h * in_w;
  std::vector<int32_t> input_int8(input_numel);

  for (int i = 0; i < input_numel; ++i) {
    input_int8[i] = quantize_input_to_int8(x_data[i], calib_scale);
  }

  // ============================================================
  // 2. conv filter: float container -> int32
  //
  // layout: [OC, IC, 1, 1]
  // per-channel scale: conv_filter_scale[oc] 只在反量化输出时使用
  // ============================================================

  std::vector<int32_t> conv_w_int8(conv_w_numel);

  for (int i = 0; i < conv_w_numel; ++i) {
    conv_w_int8[i] = load_quantized_weight(static_cast<float>(conv_w_q_float[i]));
  }

  // ============================================================
  // 3. conv2d int8_input x int8_weight -> int32 acc -> float
  //    v = acc * calib_scale * conv_filter_scale[oc] + bias
  //    -> hard_swish
  // ============================================================

  std::vector<float> conv_out_buf;
  float* conv_out = nullptr;

  int8_t* conv_out_int8 = nullptr;

  if (param.need_conv2d_output && param.conv2d_output) {
    conv_out_int8 = param.conv2d_output->mutable_data<int8_t>();
  } 
    conv_out_buf.resize(conv_out_numel);
    conv_out = conv_out_buf.data();

  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < conv_oc; ++oc) {
      const float out_scale = calib_scale * conv_filter_scale[oc];

      for (int h = 0; h < conv_out_h; ++h) {
        for (int w = 0; w < conv_out_w; ++w) {
          int32_t acc = 0;

          for (int ic = 0; ic < conv_ic; ++ic) {
            const int in_idx =
                ((n * in_c + ic) * in_h + h) * in_w + w;

            const int w_idx =
                ((oc * conv_ic + ic) * conv_kh + 0) * conv_kw + 0;

            acc += input_int8[in_idx] * conv_w_int8[w_idx];
          }

          float v = static_cast<float>(acc) * out_scale;

          if (conv_bias) {
            v += conv_bias[oc];
          }

          // 对齐 Python：反量化后直接加 bias，再 hard_swish；这里不要 round，不要 clip。
          v = hard_swish(v);

          const int out_idx =
              ((n * conv_oc + oc) * conv_out_h + h) * conv_out_w + w;

          conv_out[out_idx] = v;

          int32_t q = static_cast<int32_t>(roundf(v * (1.f / depthwise_scale)));

          if (q > 127) {
            q = 127;
          } else if (q < -127) {
            q = -127;
          }
          if (param.need_conv2d_output && param.conv2d_output){
            // std::cout << "尝试get conv2d output" << std::endl;
          conv_out_int8[out_idx] = static_cast<int8_t>(q);
          }
        }
      }
    }
  }

    // ============================================================
  // 4. conv2d float output -> depthwise input int8/int32
  // ============================================================

  std::vector<int32_t> dw_input_int8(conv_out_numel);

  for (int i = 0; i < conv_out_numel; ++i) {
    dw_input_int8[i] = quantize_input_to_int8(conv_out[i], depthwise_scale);
  }

  // ============================================================
  // 5. depthwise filter: float container -> int32
  //
  // layout: [DW_OC, 1, KH, KW]
  // per-channel scale: depthwise_filter_scale[oc] 只在反量化输出时使用
  // ============================================================

  std::vector<int32_t> dw_w_int8(dw_w_numel);

  for (int i = 0; i < dw_w_numel; ++i) {
    dw_w_int8[i] = load_quantized_weight(static_cast<float>(dw_w_q_float[i]));
  }

  // ============================================================
  // 6. depthwise params
  // ============================================================

  int stride_h = 1;
  int stride_w = 1;

  if (param.strides.size() >= 2) {
    stride_h = param.strides[0];
    stride_w = param.strides[1];
  }

  int dilation_h = 1;
  int dilation_w = 1;

  if (param.dilations && param.dilations->size() >= 2) {
    dilation_h = (*param.dilations)[0];
    dilation_w = (*param.dilations)[1];
  }

  int pad_top = 0;
  int pad_bottom = 0;
  int pad_left = 0;
  int pad_right = 0;

  if (param.paddings) {
    if (param.paddings->size() == 2) {
      pad_top = (*param.paddings)[0];
      pad_bottom = (*param.paddings)[0];
      pad_left = (*param.paddings)[1];
      pad_right = (*param.paddings)[1];
    } else if (param.paddings->size() == 4) {
      // Paddle Lite 常用顺序:
      // top, bottom, left, right
      pad_top = (*param.paddings)[0];
      pad_bottom = (*param.paddings)[1];
      pad_left = (*param.paddings)[2];
      pad_right = (*param.paddings)[3];
    }
  }

  // pad_bottom / pad_right 不直接参与索引计算，
  // out_h / out_w 已经由 output tensor dims 给出。
  (void)pad_bottom;
  (void)pad_right;

  // ============================================================
  // 7. depthwise conv2d int8_input x int8_weight -> int32 acc -> float
  //    v = acc * depthwise_scale * depthwise_filter_scale[oc] + bias
  //    -> hard_swish
  // ============================================================

  for (int n = 0; n < batch; ++n) {
    for (int oc = 0; oc < dw_oc; ++oc) {
      const float out_scale =
          depthwise_scale * depthwise_filter_scale[oc];

      // depthwise multiplier = dw_oc / groups
      // multiplier == 1 时，in_c_idx == oc
      const int in_c_idx = oc / dw_channel_multiplier;

      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int32_t acc = 0;

          for (int kh = 0; kh < dw_kh; ++kh) {
            for (int kw = 0; kw < dw_kw; ++kw) {
              const int ih =
                  oh * stride_h + kh * dilation_h - pad_top;

              const int iw =
                  ow * stride_w + kw * dilation_w - pad_left;

              if (ih < 0 || ih >= conv_out_h || iw < 0 || iw >= conv_out_w) {
                continue;
              }

              const int in_idx =
                  ((n * conv_oc + in_c_idx) * conv_out_h + ih) *
                      conv_out_w +
                  iw;

              const int w_idx =
                  ((oc * dw_ic_per_group + 0) * dw_kh + kh) * dw_kw + kw;

              acc += dw_input_int8[in_idx] * dw_w_int8[w_idx];
            }
          }

          float v = static_cast<float>(acc) * out_scale;

          if (dw_bias) {
            v += dw_bias[oc];
          }

          // 对齐 Python：反量化后直接加 bias，再 hard_swish；这里不要 round，不要 clip。
          v = hard_swish(v);

          const int out_idx =
              ((n * out_c + oc) * out_h + oh) * out_w + ow;

          out_data[out_idx] = v;
        }
      }
    }
  }
}


}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::intel_fpga::CalibConv2dCompute<PRECISION(kFloat),
                                                       PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(calib_conv2d, kIntelFPGA, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output_Conv2d", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindPaddleOpVersion("calib_conv2d", 1)
    .Finalize();

