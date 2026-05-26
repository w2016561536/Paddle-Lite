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
#include <memory>
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"
#ifdef LITE_WITH_PROFILE
#include "lite/api/paddle_place.h"
#endif

namespace paddle {
namespace lite {
namespace operators {

class CalibConv2dOpLite : public OpLite {
 public:
  CalibConv2dOpLite() {}

  explicit CalibConv2dOpLite(const std::string& type) : OpLite(type) {}

  bool CheckShape() const override;
  bool InferShapeImpl() const override;
  bool InferShapeWithCache() const override { return true; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto filter_dims = param_.filter->dims();
    auto input_dims = param_.x->dims();
    auto output_dims = param_.output->dims();
    ch->input_shape = ch->DimToStr(input_dims);
    ch->output_shape = ch->DimToStr(output_dims);
    ch->filter_shape = ch->DimToStr(filter_dims);
    ch->remark =
        std::to_string(filter_dims[2]) + "x" + std::to_string(filter_dims[3]) +
        "p" + std::to_string((*param_.paddings)[0]) + "s" +
        std::to_string(param_.strides[0]) + "g" +
        std::to_string(param_.groups) + "d" +
        std::to_string((*param_.dilations)[0]) + (param_.bias ? "Bias" : "") +
        ActivationTypeToStr(param_.activation_param.active_type);
    // MACs = 2.f * kw * kh * batchsize * out_c * out_h * out_w * in_c / group
    // GMACs = 1e-9f * MACs
    // GMACPS = 1e-6f * MACs / predict_ms
    ch->macs = 2.f * filter_dims[2] * filter_dims[3] *
               output_dims.production() * input_dims[1] / param_.groups;

    if (!param_.fuse_elementwise_op_type.empty()) {
      ch->remark += param_.fuse_elementwise_op_type;
      ch->macs += 1.0f * output_dims.production();
    }
  }
#endif

  // TODO(Superjomn) replace framework::OpDesc with a lite one.
  bool AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) override {
  // -----------------------------
  // Required inputs / outputs
  // -----------------------------
  auto input_vec = op_desc.Input("Input");
  auto filter_vec = op_desc.Input("Filter_Conv2d");
  auto dw_filter_vec = op_desc.Input("Filter_Depthwise_Conv2d");
  auto out_vec = op_desc.Output("Output");

  // CHECK(!input_vec.empty()) << "Input is empty";
  // CHECK(!filter_vec.empty()) << "Filter_Conv2d is empty";
  // CHECK(!dw_filter_vec.empty()) << "Filter_Depthwise_Conv2d is empty";
  // CHECK(!out_vec.empty()) << "Output is empty";

  auto Input = input_vec.front();
  auto Filter = filter_vec.front();
  auto Filter1 = dw_filter_vec.front();
  auto Out = out_vec.front();

  // CHECK(scope->FindVar(Input)) << "Cannot find Input var: " << Input;
  // CHECK(scope->FindVar(Filter)) << "Cannot find Filter_Conv2d var: " << Filter;
  // CHECK(scope->FindVar(Filter1))
  //     << "Cannot find Filter_Depthwise_Conv2d var: " << Filter1;
  // CHECK(scope->FindVar(Out)) << "Cannot find Output var: " << Out;

  param_.x = scope->FindVar(Input)->GetMutable<lite::Tensor>();
  param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
  param_.depthwise_filter =
      scope->FindVar(Filter1)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();

  // CHECK(param_.x);
  // CHECK(param_.filter);
  // CHECK(param_.depthwise_filter);
  // CHECK(param_.output);

  // -----------------------------
  // Optional bias inputs
  // -----------------------------
  param_.bias = nullptr;
  auto bias_vec = op_desc.Input("Bias_Conv2d");
  if (!bias_vec.empty() && !bias_vec.front().empty()) {
    auto Bias = bias_vec.front();
    // CHECK(scope->FindVar(Bias)) << "Cannot find Bias_Conv2d var: " << Bias;
    param_.bias = scope->FindVar(Bias)->GetMutable<lite::Tensor>();
    // CHECK(param_.bias);
  }

  param_.depthwise_bias = nullptr;
  auto dw_bias_vec = op_desc.Input("Bias_Depthwise_Conv2d");
  if (!dw_bias_vec.empty() && !dw_bias_vec.front().empty()) {
    auto Bias1 = dw_bias_vec.front();
    // CHECK(scope->FindVar(Bias1))
    //     << "Cannot find Bias_Depthwise_Conv2d var: " << Bias1;
    param_.depthwise_bias =
        scope->FindVar(Bias1)->GetMutable<lite::Tensor>();
    // CHECK(param_.depthwise_bias);
  }

  // -----------------------------
  // Optional conv2d_output
  // -----------------------------
  param_.need_conv2d_output = false;
  param_.conv2d_output = nullptr;

  if (op_desc.HasAttr("need_conv2d_output")) {
    param_.need_conv2d_output =
        op_desc.GetAttr<bool>("need_conv2d_output");
  }

  if (param_.need_conv2d_output) {
    auto conv_out_vec = op_desc.Output("Output_Conv2d");
    // CHECK(!conv_out_vec.empty()) << "need_conv2d_output=true, "
    //                              << "but Output_Conv2d is empty";

    auto OutConv2d = conv_out_vec.front();
    // CHECK(!OutConv2d.empty()) << "Output_Conv2d name is empty";
    // CHECK(scope->FindVar(OutConv2d))
    //     << "Cannot find Output_Conv2d var: " << OutConv2d;

    param_.conv2d_output =
        scope->FindVar(OutConv2d)->GetMutable<lite::Tensor>();
    // CHECK(param_.conv2d_output);

    output_tensor_ptrs_cache_.push_back(param_.conv2d_output);
  }

  // -----------------------------
  // Depthwise attrs
  // 注意：这些只给 depthwise_conv2d 用
  // -----------------------------
  // CHECK(op_desc.HasAttr("strides"));
  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  // CHECK_EQ(param_.strides.size(), 2);

  // CHECK(op_desc.HasAttr("paddings"));
  auto paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  // CHECK(paddings.size() == 2 || paddings.size() == 4);
  param_.paddings = std::make_shared<std::vector<int>>(paddings);

  // CHECK(op_desc.HasAttr("dilations"));
  auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  // CHECK_EQ(dilations.size(), 2);
  param_.dilations = std::make_shared<std::vector<int>>(dilations);

  // CHECK(op_desc.HasAttr("groups"));
  param_.groups = op_desc.GetAttr<int>("groups");

  // -----------------------------
  // Quant attrs
  // -----------------------------
  // CHECK(op_desc.HasAttr("calib_scale"));
  param_.calib_scale = op_desc.GetAttr<float>("calib_scale");
  // CHECK_GT(param_.calib_scale, 0.f);

  // CHECK(op_desc.HasAttr("Conv2d_Filter0_scale"));
  param_.Conv2d_Filter0_scale =
      op_desc.GetAttr<std::vector<float>>("Conv2d_Filter0_scale");
  // CHECK(!param_.Conv2d_Filter0_scale.empty());

  // CHECK(op_desc.HasAttr("depthwise_scale"));
  param_.depthwise_scale = op_desc.GetAttr<std::vector<float>>("depthwise_scale")[0];
  // CHECK_GT(param_.depthwise_scale, 0.f);

  // CHECK(op_desc.HasAttr("Depthwise2d_Filter0_scale"));
  param_.Depthwise2d_Filter0_scale =
      op_desc.GetAttr<std::vector<float>>("Depthwise2d_Filter0_scale");
  // CHECK(!param_.Depthwise2d_Filter0_scale.empty());

  if (op_desc.HasAttr("fuse_relu_before_depthwise_conv")) {
    param_.fuse_relu_before_depthwise_conv =
        op_desc.GetAttr<bool>("fuse_relu_before_depthwise_conv");
  } else {
    param_.fuse_relu_before_depthwise_conv = false;
  }

  // -----------------------------
  // Cache tensors
  // -----------------------------
  input_tensor_ptrs_cache_.push_back(param_.x);
  input_tensor_ptrs_cache_.push_back(param_.filter);
  input_tensor_ptrs_cache_.push_back(param_.depthwise_filter);

  if (param_.bias) {
    input_tensor_ptrs_cache_.push_back(param_.bias);
  }

  if (param_.depthwise_bias) {
    input_tensor_ptrs_cache_.push_back(param_.depthwise_bias);
  }

  output_tensor_ptrs_cache_.push_back(param_.output);

  return true;
}

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "conv2d"; }

 protected:
  mutable ConvParam param_;
  std::string padding_algorithm_{""};
};
// update padding dilation
void UpdatePaddingAndDilation(std::vector<int>* paddings,
                              std::vector<int>* dilations,
                              const std::vector<int>& strides,
                              const std::string padding_algorithm,
                              const lite::DDim data_dims,
                              const lite::DDim& ksize);
}  // namespace operators
}  // namespace lite
}  // namespace paddle
