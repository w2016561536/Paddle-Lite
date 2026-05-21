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

#include "lite/core/optimizer/mir/subgraph/fpga_conv_fuser_with_branch.h"
#include <cmath>
#include <memory>
#include <vector>
#include <iostream>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FpgaConvFuserWithBranch::BuildPattern() { 

  auto inputs_teller1 = [](const Node* node) -> bool { 
    // Conv2d + Bias 的输入输出条件，要求filter必须满足m*n*1*1
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto input_weight_name = op_desc.Input("Filter").front();
    auto input_has_bias = op_desc.HasInput("Bias") && !(op_desc.Input("Bias").empty());
    if (!input_has_bias) {
      std::cout << "conv2d has no bias input" << std::endl;
      return false;
    }
    auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
    auto w_shape = scope->FindVar(input_weight_name)->Get<lite::Tensor>().dims();
    size_t w_rank = w_shape.size();
    auto res = (w_rank == 4 && w_shape[2] == 1 && w_shape[3] == 1);
    std::cout << "input weight shape: " << w_shape.repr() << ", res: " << res
              << std::endl;
    int bit_length = op_desc.GetAttr<int>("bit_length");
    bool enable_int8 = op_desc.GetAttr<bool>("enable_int8");
    std::vector<int> dilations = op_desc.GetAttr<std::vector<int>>("dilations");
    int groups = op_desc.GetAttr<int>("groups");
        
    bool op_is_conv2d = true;
    if (bit_length != 8){
      op_is_conv2d = false;
      std::cout << "bit_length: " << bit_length << std::endl;
    }
    if (!enable_int8) {
      op_is_conv2d = false;
      std::cout << "enable_int8: " << enable_int8 << std::endl;
    }
    for (auto dilation : dilations) {
      if (dilation != 1) {
        op_is_conv2d = false;
          std::cout << "dilation: " << dilation << std::endl;
        break;
      }
    }
    if (groups != 1) {
      op_is_conv2d = false;
      std::cout << "groups: " << groups << std::endl;
    }
    if (op_desc.HasAttr("act_type")) { // 可能不需要激活
      std::string act_type = op_desc.GetAttr<std::string>("act_type");
      if (act_type == "hard_swish") {
      float hard_swish_offset = op_desc.GetAttr<float>("hard_swish_offset");
      float hard_swish_threshold = op_desc.GetAttr<float>("hard_swish_threshold");
      float hard_swish_scale = op_desc.GetAttr<float>("hard_swish_scale");

      if (std::fabs(hard_swish_offset - 3) > 1e-5 ||
        std::fabs(hard_swish_threshold - 6) > 1e-5 ||
        std::fabs(hard_swish_scale - 6) > 1e-5) {
        op_is_conv2d = false;
        std::cout << "hard_swish_offset: " << hard_swish_offset << ", hard_swish_threshold: " << hard_swish_threshold
                  << ", hard_swish_scale: " << hard_swish_scale << std::endl;
        }
      }else{
        op_is_conv2d = false;
        std::cout << "act_type: " << act_type << std::endl;
      }
    }else{
       return false;
    }
    if (!res){
      op_is_conv2d = false;
      std::cout << "filter shape is not 4D with 1x1 spatial dimensions" << std::endl;
    }
    return op_is_conv2d;
    // return res;
  };

  auto input_attr_teller = [](const Node* node) -> bool {
    // hard swish 的attr条件，scale = 6, offset = 3, threshold = 6
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    if (!op_desc.HasAttr("offset")) {
      std::cout << "hard swish no offset attr" << std::endl;
      return false;
    }
    if (!op_desc.HasAttr("threshold")) {
      std::cout << "hard swish no threshold attr" << std::endl;
      return false;
    }
    if (!op_desc.HasAttr("scale")) {
      std::cout << "hard swish no scale attr" << std::endl;
      return false;
    }
    float offset = op_desc.GetAttr<float>("offset");
    float threshold = op_desc.GetAttr<float>("threshold");
    float scale = op_desc.GetAttr<float>("scale");
    bool res = (std::fabs(offset - 3) < 1e-5
                && std::fabs(threshold - 6) < 1e-5
                && std::fabs(scale - 6) < 1e-5);
    std::cout << "hard swish offset: " << offset << ", threshold: "
              << threshold << ", scale: " << scale << ", res: " << res
              << std::endl;
    return res;
  };

  // create nodes.
  auto* calibInput = VarNode("Input");
  auto* calibOutput = VarNode("CalibOutput"); // 实际上就是conv的输入，与convInput等价

  auto* CalibNode = OpNode("calib", "calib");
  
  // auto* convInput = VarNode("Input")->assert_is_op_input("conv2d", "Input");
  auto* convNode = OpNode("conv2d", "conv2d")->assert_node_satisfied(inputs_teller1);
  auto* convFilter = VarNode("ConvFilter");
  auto* convBias = VarNode("ConvBias");
  // auto* Filter0_scale = VarNode("Filter0_scale")->assert_is_persistable_var();
  auto* convOutput = VarNode("ConvOutput");

  auto* depthwise2dConvNode = OpNode("depthwise_conv2d", "depthwise_conv2d");
  auto* depthwise2dConvFilter = VarNode("DepthwiseFilter");
  auto* depthwise2dConvBias = VarNode("DepthwiseBias");
  auto* depthwise2dConvOutput = VarNode("DepthwiseOutput");

  auto* hard_swishNode = OpNode("hard_swish", "hard_swish") ->assert_node_satisfied(input_attr_teller);
  auto* hard_swishOutput = VarNode("Output");

  // create topology.
  std::vector<PMNode*> CalibNode_input{calibInput};
  std::vector<PMNode*> Conv2dNode_input{calibOutput, convFilter, convBias};
  std::vector<PMNode*> DepthwiseConv2dNode_input{convOutput, depthwise2dConvFilter, depthwise2dConvBias};
  std::vector<PMNode*> HardSwishNode_input{depthwise2dConvOutput};
  // mul_inputs >> *mul >> *mul_out;
  CalibNode_input >> *CalibNode >> *calibOutput;

  // Some op specialities.
  calibOutput->AsIntermediate();
  convNode->AsIntermediate();
  CalibNode->AsIntermediate();

  Conv2dNode_input >> *convNode >> *convOutput;

  DepthwiseConv2dNode_input >> *depthwise2dConvNode >> *depthwise2dConvOutput;
  HardSwishNode_input >> *hard_swishNode >> *hard_swishOutput;

  convOutput->AsOutput(); // 这里，要求也输出conv2d的结果
  depthwise2dConvOutput->AsIntermediate();
  depthwise2dConvNode->AsIntermediate();
  hard_swishNode->AsIntermediate();
}

void FpgaConvFuserWithBranch::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  std::cout << "insert new node for conv2d" << std::endl;
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
  std::cout << "valid_places size: " << valid_places.size() << std::endl;
  std::cout << "scope: " << scope << std::endl;
  fc_op->Attach(op_desc, scope);

  std::cout << "Attach new conv2d op: " << fc_op->DebugString() << std::endl;

  auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

  IR_NODE_LINK_TO(matched.at("Input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("ConvFilter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("ConvBias"), new_op_node);
  IR_NODE_LINK_TO(matched.at("DepthwiseFilter"), new_op_node)
  IR_NODE_LINK_TO(matched.at("DepthwiseBias"), new_op_node)

  IR_NODE_LINK_TO(new_op_node, matched.at("Output"));
  IR_NODE_LINK_TO(new_op_node, matched.at("ConvOutput"));
}

cpp::OpDesc FpgaConvFuserWithBranch::GenOpDesc(const key2nodes_t& matched) {
  auto* conv_info = matched.at("conv2d")->stmt()->op_info();
  auto* calib_info = matched.at("calib")->stmt()->op_info();
  auto* depthwise_conv_info = matched.at("depthwise_conv2d")->stmt()->op_info();
  auto* hard_swish_info = matched.at("hard_swish")->stmt()->op_info();

  cpp::OpDesc op_desc;
  op_desc.SetType("calib_conv2d");

  // 1. 输入应当来自 calib 的输入，而不是 conv2d 的输入
  // 因为融合前结构一般是：Input -> calib -> conv2d -> Output
  op_desc.SetInput("Input", calib_info->Input("Input"));

  // 2. conv2d 的权重、bias 保持原来的变量名
  op_desc.SetInput("Filter_Conv2d", conv_info->Input("Filter"));

  if (conv_info->HasInput("Bias") && !conv_info->Input("Bias").empty()) {
    op_desc.SetInput("Bias_Conv2d", conv_info->Input("Bias"));
  }

  op_desc.SetInput("Filter_Depthwise_Conv2d", depthwise_conv_info->Input("Filter"));
  op_desc.SetInput("Bias_Depthwise_Conv2d", depthwise_conv_info->Input("Bias"));

  // 3. 输出使用 conv2d 的最终输出
  op_desc.SetOutput("Output", hard_swish_info->Output("Out"));
  op_desc.SetAttr("need_conv2d_output", true);
  op_desc.SetOutput("Output_Conv2d",conv_info->Output("Output"));


  // 如果 calib 的 scale 是 attr，而不是 input，也要保留
  if (calib_info->HasAttr("scale")) {
    op_desc.SetAttr("calib_scale", calib_info->GetAttr<float>("scale"));
  }

  // 5. 保留 conv2d 属性
  if (conv_info->HasAttr("Filter0_scale")) {
    op_desc.SetAttr(
        "Conv2d_Filter0_scale",
        conv_info->GetAttr<std::vector<float>>("Filter0_scale"));
  }

  if (depthwise_conv_info->HasAttr("Filter0_scale")) {
    op_desc.SetAttr(
        "Depthwise2d_Filter0_scale",
        depthwise_conv_info->GetAttr<std::vector<float>>("Filter0_scale"));
  }

  if (depthwise_conv_info->HasAttr("Input0_scale")) {
    op_desc.SetAttr("depthwise_scale", depthwise_conv_info->GetAttr<std::vector<float>>("Input0_scale"));
  }

  op_desc.SetAttr("dilations", depthwise_conv_info->GetAttr<std::vector<int>>("dilations"));
  op_desc.SetAttr("groups", depthwise_conv_info->GetAttr<int>("groups"));
  op_desc.SetAttr("padding_algorithm",
                  depthwise_conv_info->GetAttr<std::string>("padding_algorithm"));

  op_desc.SetAttr("strides", depthwise_conv_info->GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings", depthwise_conv_info->GetAttr<std::vector<int>>("paddings"));
  

  return op_desc;
}


}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
