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

#include "lite/core/optimizer/mir/subgraph/fpga_conv_fuser.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

namespace {

bool IsHardSwishAttr(const cpp::OpDesc& op_desc,
                     const std::string& offset_name,
                     const std::string& threshold_name,
                     const std::string& scale_name) {
  if (!op_desc.HasAttr(offset_name) || !op_desc.HasAttr(threshold_name) ||
      !op_desc.HasAttr(scale_name)) {
    return false;
  }
  const float offset = op_desc.GetAttr<float>(offset_name);
  const float threshold = op_desc.GetAttr<float>(threshold_name);
  const float scale = op_desc.GetAttr<float>(scale_name);
  return std::fabs(offset - 3.f) < 1e-5f &&
         std::fabs(threshold - 6.f) < 1e-5f &&
         std::fabs(scale - 6.f) < 1e-5f;
}

bool IsSupportedConv1x1Int8(const Node* node) {
  auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
  const auto input_weight_name = op_desc.Input("Filter").front();
  const bool input_has_bias =
      op_desc.HasInput("Bias") && !op_desc.Input("Bias").empty();
  if (!input_has_bias) {
    std::cout << "conv2d has no bias input" << std::endl;
    return false;
  }

  auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
  auto w_shape = scope->FindVar(input_weight_name)->Get<lite::Tensor>().dims();
  const bool is_1x1 = w_shape.size() == 4 && w_shape[2] == 1 && w_shape[3] == 1;
  if (!is_1x1) {
    std::cout << "filter shape is not 4D with 1x1 spatial dimensions: "
              << w_shape.repr() << std::endl;
    return false;
  }

  if (!op_desc.HasAttr("bit_length") ||
      op_desc.GetAttr<int>("bit_length") != 8) {
    std::cout << "conv2d bit_length is not 8" << std::endl;
    return false;
  }
  if (!op_desc.HasAttr("enable_int8") ||
      !op_desc.GetAttr<bool>("enable_int8")) {
    std::cout << "conv2d enable_int8 is false" << std::endl;
    return false;
  }

  const auto dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  for (auto dilation : dilations) {
    if (dilation != 1) {
      std::cout << "conv2d dilation is not 1: " << dilation << std::endl;
      return false;
    }
  }
  if (op_desc.GetAttr<int>("groups") != 1) {
    std::cout << "conv2d groups is not 1" << std::endl;
    return false;
  }

  if (op_desc.HasAttr("act_type")) {
    const auto act_type = op_desc.GetAttr<std::string>("act_type");
    if (act_type.empty()) {
      return true;
    }
    if (act_type != "hard_swish") {
      std::cout << "conv2d act_type is unsupported: " << act_type << std::endl;
      return false;
    }
    if (!IsHardSwishAttr(op_desc,
                         "hard_swish_offset",
                         "hard_swish_threshold",
                         "hard_swish_scale")) {
      std::cout << "conv2d hard_swish attrs are unsupported" << std::endl;
      return false;
    }
  }

  return true;
}

bool IsSupportedHardSwish(const Node* node) {
  auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
  if (!IsHardSwishAttr(op_desc, "offset", "threshold", "scale")) {
    std::cout << "hard_swish attrs are unsupported" << std::endl;
    return false;
  }
  return true;
}

float FirstScaleAttr(const cpp::OpDesc* op_info,
                     const std::string& attr_name,
                     float fallback) {
  if (!op_info->HasAttr(attr_name)) {
    return fallback;
  }
  const auto scales = op_info->GetAttr<std::vector<float>>(attr_name);
  return scales.empty() ? fallback : scales[0];
}

std::vector<float> ScaleVector(const std::vector<float>& values,
                               float multiplier) {
  std::vector<float> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i] * multiplier;
  }
  return out;
}

}  // namespace

void FpgaConvFuser::BuildPattern() {
  auto* input = VarNode("Input");

  auto* conv_node =
      OpNode("conv2d", "conv2d")->assert_node_satisfied(IsSupportedConv1x1Int8);
  auto* conv_filter = VarNode("ConvFilter")->assert_is_persistable_var();
  auto* conv_bias = VarNode("ConvBias")->assert_is_persistable_var();
  auto* conv_output = VarNode("ConvOutput")->assert_only_one_output();

  auto* depthwise_node = OpNode("depthwise_conv2d", "depthwise_conv2d");
  auto* depthwise_filter =
      VarNode("DepthwiseFilter")->assert_is_persistable_var();
  auto* depthwise_bias = VarNode("DepthwiseBias")->assert_is_persistable_var();
  auto* depthwise_output =
      VarNode("DepthwiseOutput")->assert_only_one_output();

  auto* hard_swish_node =
      OpNode("hard_swish", "hard_swish")->assert_node_satisfied(
          IsSupportedHardSwish);
  auto* hard_swish_output =
      VarNode("HardSwishOutput")->assert_only_one_output();

  auto* calib_node = OpNode("calib", "calib");
  auto* output = VarNode("Output");

  input->AsInput();
  conv_node->AsIntermediate();
  conv_output->AsIntermediate();
  depthwise_node->AsIntermediate();
  depthwise_output->AsIntermediate();
  hard_swish_node->AsIntermediate();
  hard_swish_output->AsIntermediate();
  calib_node->AsIntermediate();
  output->AsOutput();

  std::vector<PMNode*> conv_inputs{input, conv_filter, conv_bias};
  std::vector<PMNode*> depthwise_inputs{
      conv_output, depthwise_filter, depthwise_bias};
  std::vector<PMNode*> hardswish_inputs{depthwise_output};
  std::vector<PMNode*> calib_inputs{hard_swish_output};

  conv_inputs >> *conv_node >> *conv_output;
  depthwise_inputs >> *depthwise_node >> *depthwise_output;
  hardswish_inputs >> *hard_swish_node >> *hard_swish_output;
  calib_inputs >> *calib_node >> *output;
}

void FpgaConvFuser::InsertNewNode(SSAGraph* graph,
                                  const key2nodes_t& matched) {
  std::cout << "insert new node for calib_conv2d" << std::endl;
  auto conv_op = matched.at("conv2d")->stmt()->op();
  auto* scope = conv_op->scope();

  auto op_desc = GenOpDesc(matched);
  auto fc_op = LiteOpRegistry::Global().Create("calib_conv2d");
  if (!fc_op) {
    LOG(FATAL) << "Create calib_conv2d failed. "
               << "Check REGISTER_LITE_OP(calib_conv2d, ...) "
               << "and whether the source file is compiled/linked.";
  }

  auto& valid_places = conv_op->valid_places();
  std::cout << "op_desc " << op_desc.Repr() << std::endl;
  fc_op->Attach(op_desc, scope);
  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("Input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("ConvFilter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("ConvBias"), new_op_node);
  IR_NODE_LINK_TO(matched.at("DepthwiseFilter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("DepthwiseBias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("Output"));
}

cpp::OpDesc FpgaConvFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* conv_info = matched.at("conv2d")->stmt()->op_info();
  auto* calib_info = matched.at("calib")->stmt()->op_info();
  auto* depthwise_info = matched.at("depthwise_conv2d")->stmt()->op_info();

  cpp::OpDesc op_desc;
  op_desc.SetType("calib_conv2d");
  op_desc.SetInput("Input", conv_info->Input("Input"));
  op_desc.SetInput("Filter_Conv2d", conv_info->Input("Filter"));
  if (conv_info->HasInput("Bias") && !conv_info->Input("Bias").empty()) {
    op_desc.SetInput("Bias_Conv2d", conv_info->Input("Bias"));
  }
  op_desc.SetInput("Filter_Depthwise_Conv2d", depthwise_info->Input("Filter"));
  op_desc.SetInput("Bias_Depthwise_Conv2d", depthwise_info->Input("Bias"));
  op_desc.SetOutput("Output", calib_info->Output("Out"));

  const float input_scale = FirstScaleAttr(conv_info, "Input0_scale", 1.f);
  const float output_scale =
      calib_info->HasAttr("scale") ? calib_info->GetAttr<float>("scale") : 1.f;

  op_desc.SetAttr("calib_scale", output_scale);

  if (conv_info->HasAttr("Filter0_scale")) {
    const auto conv_filter_scale =
        conv_info->GetAttr<std::vector<float>>("Filter0_scale");
    op_desc.SetAttr("Conv2d_Filter0_scale",
                    ScaleVector(conv_filter_scale, input_scale / output_scale));
  }
  if (depthwise_info->HasAttr("Filter0_scale")) {
    op_desc.SetAttr("Depthwise2d_Filter0_scale",
                    depthwise_info->GetAttr<std::vector<float>>(
                        "Filter0_scale"));
  }
  if (depthwise_info->HasAttr("Input0_scale")) {
    op_desc.SetAttr("depthwise_scale",
                    depthwise_info->GetAttr<std::vector<float>>(
                        "Input0_scale"));
  }

  op_desc.SetAttr("dilations",
                  depthwise_info->GetAttr<std::vector<int>>("dilations"));
  op_desc.SetAttr("groups", depthwise_info->GetAttr<int>("groups"));
  op_desc.SetAttr("padding_algorithm",
                  depthwise_info->GetAttr<std::string>("padding_algorithm"));
  op_desc.SetAttr("strides",
                  depthwise_info->GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings",
                  depthwise_info->GetAttr<std::vector<int>>("paddings"));
  op_desc.SetAttr("need_conv2d_output", false);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
