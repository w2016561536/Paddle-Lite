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
    auto Input = op_desc.Input("Input").front();
    auto Filter = op_desc.Input("Filter_Conv2d").front();
    auto Filter1 = op_desc.Input("Filter_Depthwise_Conv2d").front();
    auto Bias = op_desc.Input("Bias_Conv2d").front();
    auto Bias1 = op_desc.Input("Bias_Depthwise_Conv2d").front();
    auto Out = op_desc.Output("Output").front();

    if (op_desc.HasAttr("need_conv2d_output") && op_desc.GetAttr<bool>("need_conv2d_output") == true){
      auto Out_conv2d = op_desc.Output("Output_Conv2d").front();
      param_.conv2d_output = scope->FindVar(Out_conv2d)->GetMutable<lite::Tensor>();
      output_tensor_ptrs_cache_.push_back(param_.conv2d_output);
    }

    param_.x = scope->FindVar(Input)->GetMutable<lite::Tensor>();
    param_.filter = scope->FindVar(Filter)->GetMutable<lite::Tensor>();
    param_.depthwise_filter = scope->FindVar(Filter1)->GetMutable<lite::Tensor>();
    param_.bias = scope->FindVar(Bias)->GetMutable<lite::Tensor>();
    param_.depthwise_bias = scope->FindVar(Bias1)->GetMutable<lite::Tensor>();
    param_.output = scope->FindVar(Out)->GetMutable<lite::Tensor>();
    // CHECK(param_.x);
    // CHECK(param_.filter);
    // CHECK(param_.output);
    input_tensor_ptrs_cache_.push_back(param_.x);
    output_tensor_ptrs_cache_.push_back(param_.output);

    param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
    std::vector<int> paddings = op_desc.GetAttr<std::vector<int>>("paddings");
    param_.groups = op_desc.GetAttr<int>("groups");
    auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    param_.dilations = std::make_shared<std::vector<int>>(dilations);

    param_.paddings = std::make_shared<std::vector<int>>(paddings);
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
