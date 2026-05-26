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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

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
  #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::cout << "USE ARM NEON FOR ACCEL" << std::endl;
  #endif

  // impl_->SetContext(std::move(arm_cxt_));
  // impl_->SetParam(param);
  // impl_->PrepareForRun();

}
// 如果当前源文件尚未包含这些头文件，请在文件头补充：
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <vector>
// #if defined(__ARM_NEON) || defined(__ARM_NEON__)
// #include <arm_neon.h>
// #endif

template <>
void CalibConv2dCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();

  // ------------------------------------------------------------
  // Tensor data
  // ------------------------------------------------------------

  const float* x_data = param.x->data<float>();

  // filter tensor 当前按 int8_t 读取：其数值语义应为量化后的 int8 权重。
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
  (void)out_n;

  const int dw_channel_multiplier = dw_oc / param.groups;

  const float calib_scale = param.calib_scale;
  const float depthwise_scale = param.depthwise_scale;

  const auto& conv_filter_scale = param.Conv2d_Filter0_scale;
  const auto& depthwise_filter_scale = param.Depthwise2d_Filter0_scale;

  const int conv_w_numel = conv_oc * conv_ic * conv_kh * conv_kw;
  const int dw_w_numel = dw_oc * dw_ic_per_group * dw_kh * dw_kw;

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

  auto round_up_16 = [](int x) -> int {
    return (x + 15) & ~15;
  };

  auto quantize_input_to_int8 = [](float x, float scale) -> int8_t {
    // 对齐 Paddle Lite calib:
    // q = roundf(x * (1.f / scale))
    // clip 到 [-127, 127]
    int32_t q = static_cast<int32_t>(roundf(x * (1.f / scale)));

    if (q > 127) {
      q = 127;
    } else if (q < -127) {
      q = -127;
    }

    return static_cast<int8_t>(q);
  };

  auto quantize_float_array_to_int8 =
      [&](const float* src, int8_t* dst, int numel, float scale) {
    const float inv_scale = 1.f / scale;
    int i = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // ARMv7-A/ARMv7HF NEON 路径：
    // 1) float * inv_scale
    // 2) 用 +/-0.5 + vcvt 实现与 roundf 接近的四舍五入到整数
    // 3) clip 到 [-127, 127]
    // 4) narrow 到 int8_t
    const float32x4_t vinv = vdupq_n_f32(inv_scale);
    const float32x4_t vzero_f32 = vdupq_n_f32(0.f);
    const float32x4_t vhalf_f32 = vdupq_n_f32(0.5f);
    const int32x4_t vmax_s32 = vdupq_n_s32(127);
    const int32x4_t vmin_s32 = vdupq_n_s32(-127);

    for (; i + 8 <= numel; i += 8) {
      float32x4_t v0 = vmulq_f32(vld1q_f32(src + i), vinv);
      float32x4_t v1 = vmulq_f32(vld1q_f32(src + i + 4), vinv);

      const uint32x4_t m0 = vcgeq_f32(v0, vzero_f32);
      const uint32x4_t m1 = vcgeq_f32(v1, vzero_f32);

      v0 = vbslq_f32(m0, vaddq_f32(v0, vhalf_f32),
                         vsubq_f32(v0, vhalf_f32));
      v1 = vbslq_f32(m1, vaddq_f32(v1, vhalf_f32),
                         vsubq_f32(v1, vhalf_f32));

      int32x4_t q0 = vcvtq_s32_f32(v0);
      int32x4_t q1 = vcvtq_s32_f32(v1);

      q0 = vmaxq_s32(vminq_s32(q0, vmax_s32), vmin_s32);
      q1 = vmaxq_s32(vminq_s32(q1, vmax_s32), vmin_s32);

      const int16x4_t q16_0 = vmovn_s32(q0);
      const int16x4_t q16_1 = vmovn_s32(q1);
      const int16x8_t q16 = vcombine_s16(q16_0, q16_1);
      const int8x8_t q8 = vmovn_s16(q16);

      vst1_s8(dst + i, q8);
    }
#endif

    for (; i < numel; ++i) {
      dst[i] = quantize_input_to_int8(src[i], scale);
    }
  };

  auto clear_nhwc_c_padding = [](int8_t* dst,
                                 int pixel_count,
                                 int c,
                                 int c_aligned) {
    const int pad = c_aligned - c;
    if (pad <= 0) {
      return;
    }

    for (int p = 0; p < pixel_count; ++p) {
      std::memset(dst + p * c_aligned + c, 0, pad * sizeof(int8_t));
    }
  };

  auto quantize_nchw_float_to_nhwc_int8_c16 =
      [&](const float* src,
          int8_t* dst,
          int n_num,
          int c_num,
          int h_num,
          int w_num,
          int c_aligned,
          float scale) {
    const int spatial = h_num * w_num;
    const int pixel_count = n_num * spatial;
    const float inv_scale = 1.f / scale;

    // 只清零 C 维补齐区，不再对整个 NHWC buffer 做 memset。
    // 实际通道 [0, c_num) 会在下面的融合量化重排循环中全部写满。
    clear_nhwc_c_padding(dst, pixel_count, c_num, c_aligned);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    const float32x4_t vinv = vdupq_n_f32(inv_scale);
    const float32x4_t vzero_f32 = vdupq_n_f32(0.f);
    const float32x4_t vhalf_f32 = vdupq_n_f32(0.5f);
    const int32x4_t vmax_s32 = vdupq_n_s32(127);
    const int32x4_t vmin_s32 = vdupq_n_s32(-127);

#define STORE_NHWC_8_LANES(q8, out_ptr, stride)        \
    do {                                                \
      (out_ptr)[0 * (stride)] = vget_lane_s8((q8), 0);  \
      (out_ptr)[1 * (stride)] = vget_lane_s8((q8), 1);  \
      (out_ptr)[2 * (stride)] = vget_lane_s8((q8), 2);  \
      (out_ptr)[3 * (stride)] = vget_lane_s8((q8), 3);  \
      (out_ptr)[4 * (stride)] = vget_lane_s8((q8), 4);  \
      (out_ptr)[5 * (stride)] = vget_lane_s8((q8), 5);  \
      (out_ptr)[6 * (stride)] = vget_lane_s8((q8), 6);  \
      (out_ptr)[7 * (stride)] = vget_lane_s8((q8), 7);  \
    } while (0)
#endif

    for (int n = 0; n < n_num; ++n) {
      const float* src_n = src + n * c_num * spatial;
      int8_t* dst_n = dst + n * spatial * c_aligned;

      for (int ic = 0; ic < c_num; ++ic) {
        const float* src_plane = src_n + ic * spatial;
        int8_t* dst_channel = dst_n + ic;
        int s = 0;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        // NCHW 中同一通道的 H*W 平面是连续的，因此沿空间维度做 NEON 量化；
        // 量化结果直接散写到 NHWC/C16：dst[(n, h, w, ic)]。
        for (; s + 8 <= spatial; s += 8) {
#if defined(__GNUC__)
          __builtin_prefetch(src_plane + s + 32);
#endif
          float32x4_t v0 = vmulq_f32(vld1q_f32(src_plane + s), vinv);
          float32x4_t v1 = vmulq_f32(vld1q_f32(src_plane + s + 4), vinv);

          const uint32x4_t m0 = vcgeq_f32(v0, vzero_f32);
          const uint32x4_t m1 = vcgeq_f32(v1, vzero_f32);

          v0 = vbslq_f32(m0, vaddq_f32(v0, vhalf_f32),
                             vsubq_f32(v0, vhalf_f32));
          v1 = vbslq_f32(m1, vaddq_f32(v1, vhalf_f32),
                             vsubq_f32(v1, vhalf_f32));

          int32x4_t q0 = vcvtq_s32_f32(v0);
          int32x4_t q1 = vcvtq_s32_f32(v1);

          q0 = vmaxq_s32(vminq_s32(q0, vmax_s32), vmin_s32);
          q1 = vmaxq_s32(vminq_s32(q1, vmax_s32), vmin_s32);

          const int16x4_t q16_0 = vmovn_s32(q0);
          const int16x4_t q16_1 = vmovn_s32(q1);
          const int16x8_t q16 = vcombine_s16(q16_0, q16_1);
          const int8x8_t q8 = vmovn_s16(q16);

          int8_t* dst_ptr = dst_channel + s * c_aligned;
          STORE_NHWC_8_LANES(q8, dst_ptr, c_aligned);
        }
#endif

        for (; s < spatial; ++s) {
          dst_channel[s * c_aligned] = quantize_input_to_int8(src_plane[s], scale);
        }
      }
    }

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#undef STORE_NHWC_8_LANES
#endif
  };

  // ============================================================
  // 1. x_data: float NCHW -> int8 NHWC(C16 padded)
  //
  // 当前版本将量化和 NCHW->NHWC 重排融合在一个 pass 中完成，
  // 不再生成 input_nchw_int8 临时数组，从而减少一次完整输入张量的写入和读取。
  // NHWC 的 C 维向上补齐到 16，补齐位置填 0；同时让 NHWC buffer
  // 起始地址 16 字节对齐，方便后续硬件以 16B 向量读取每个像素的通道数据。
  // ============================================================

  const int in_c_aligned = round_up_16(in_c);
  const int input_nhwc_numel = batch * in_h * in_w * in_c_aligned;

  // 额外分配 16 字节，手动把 data 指针抬到 16B 对齐位置。
  std::vector<int8_t> input_nhwc_storage(input_nhwc_numel + 16);
  const uintptr_t raw_addr =
      reinterpret_cast<uintptr_t>(input_nhwc_storage.data());
  int8_t* input_nhwc_int8 = reinterpret_cast<int8_t*>(
      (raw_addr + 15u) & ~static_cast<uintptr_t>(15u));
      // 这里的目的是让地址按16字节对齐, DRAM实现的时候最好按照物理地址对其16字节
  // int8_t* input_nhwc_int8 = reinterpret_cast<int8_t*>(raw_addr);

  quantize_nchw_float_to_nhwc_int8_c16(x_data,
                                       input_nhwc_int8,
                                       batch,
                                       in_c,
                                       in_h,
                                       in_w,
                                       in_c_aligned,
                                       calib_scale);

  // ============================================================
  // 2. conv filter: int8 container -> int8
  //
  // layout: [OC, IC, 1, 1]
  // per-channel scale: conv_filter_scale[oc] 只在反量化输出时使用
  // ============================================================

  std::vector<int8_t> conv_w_int8(conv_w_numel);

  for (int i = 0; i < conv_w_numel; ++i) {
    conv_w_int8[i] = conv_w_q_float[i];
  }

  // ============================================================
  // 3. conv2d int8_input(NHWC C16 padded) x int8_weight -> int32 acc -> float
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

          const int nhwc_base =
              ((n * conv_out_h + h) * conv_out_w + w) * in_c_aligned;

          for (int ic = 0; ic < conv_ic; ++ic) {
            const int in_idx = nhwc_base + ic;

            const int w_idx =
                ((oc * conv_ic + ic) * conv_kh + 0) * conv_kw + 0;

            const int32_t x_q = static_cast<int32_t>(input_nhwc_int8[in_idx]);
            const int32_t w_q = static_cast<int32_t>(conv_w_int8[w_idx]);
            acc += x_q * w_q;
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
          if (param.need_conv2d_output && param.conv2d_output) {
            conv_out_int8[out_idx] = static_cast<int8_t>(q);
          }
        }
      }
    }
  }

  // ============================================================
  // 4. conv2d float output -> depthwise input int8
  //
  // 后续软件模拟不额外做 NHWC/C16 优化，保持 NCHW 下标；
  // 计算时再将 int8 显式转换为 int32。
  // ============================================================

  std::vector<int8_t> dw_input_int8(conv_out_numel);
  quantize_float_array_to_int8(conv_out,
                               dw_input_int8.data(),
                               conv_out_numel,
                               depthwise_scale);

  // ============================================================
  // 5. depthwise filter: int8 container -> int8
  //
  // layout: [DW_OC, 1, KH, KW]
  // per-channel scale: depthwise_filter_scale[oc] 只在反量化输出时使用
  // ============================================================

  std::vector<int8_t> dw_w_int8(dw_w_numel);

  for (int i = 0; i < dw_w_numel; ++i) {
    dw_w_int8[i] = dw_w_q_float[i];
  }

  // ============================================================
  // 6. depthwise params
  // ============================================================

  int stride_h = 1;
  int stride_w = 1;

  stride_h = param.strides[0];
  stride_w = param.strides[1];

  int dilation_h = 1;
  int dilation_w = 1;

  // if (param.dilations && param.dilations->size() >= 2) {
  //   dilation_h = (*param.dilations)[0];
  //   dilation_w = (*param.dilations)[1];
  // }

  int pad_top = 0;
  int pad_bottom = 0;
  int pad_left = 0;
  int pad_right = 0;

  pad_top = (*param.paddings)[0];
  pad_bottom = (*param.paddings)[0];
  pad_left = (*param.paddings)[1];
  pad_right = (*param.paddings)[1];

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

              const int32_t x_q = static_cast<int32_t>(dw_input_int8[in_idx]);
              const int32_t w_q = static_cast<int32_t>(dw_w_int8[w_idx]);
              acc += x_q * w_q;
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

