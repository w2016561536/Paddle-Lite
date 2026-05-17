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
#include <memory>
#include <vector>
#include <iostream>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void FpgaConvFuser::BuildPattern() { 
  // calib 匹配条件，但是CALIB不需要求输入输出的shape，所以不需要inputs_teller和input_attr_teller
  // auto inputs_teller0 = [](const Node* node) -> bool {
  //   auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
  //   auto input_w_name = op_desc.Input("Y").front();
  //   auto* scope = const_cast<Node*>(node)->AsStmt().op()->scope();
  //   auto w_shape = scope->FindVar(input_w_name)->Get<lite::Tensor>().dims();
  //   size_t w_rank = w_shape.size();
  //   bool res = w_rank == 2;
  //   return res;
  // };

  auto inputs_teller1 = [](const Node* node) -> bool { 
    // Conv2d + Bias 的输入输出条件，要求filter必须满足m*n*1*1
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    auto input_weight_name = op_desc.Input("Filter").front();
    auto input_has_bias = op_desc.HasInput("Bias");
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
    }
    if (!res){
      op_is_conv2d = false;
      std::cout << "filter shape is not 4D with 1x1 spatial dimensions" << std::endl;
    }
    return op_is_conv2d;
    // return res;
  };

  auto input_attr_teller = [](const Node* node) -> bool {
    // calib 的attr条件，有scale
    auto op_desc = *const_cast<Node*>(node)->stmt()->op_info();
    if (op_desc.HasAttr("scale")) {
      // std::cout << "has scale attr" << std::endl;
      return true;
    }
    return false;
  };

  // create nodes.
  auto* calibInput = VarNode("Input");
  auto* calibOutput = VarNode("Out"); // 实际上就是conv的输入，与convInput等价

  auto* CalibNode = OpNode("calib", "calib");
  
  // auto* convInput = VarNode("Input")->assert_is_op_input("conv2d", "Input");
  auto* convNode = OpNode("conv2d", "conv2d")->assert_node_satisfied(inputs_teller1);
  auto* convFilter = VarNode("Filter");
  auto* convBias = VarNode("Bias");
  // auto* Filter0_scale = VarNode("Filter0_scale")->assert_is_persistable_var();
  auto* convOutput = VarNode("Output");


  // auto* x = VarNode("x")->assert_is_op_input(op_type_, "X");
  // auto* W = VarNode("W")->assert_is_op_input(op_type_, "Y");
  // auto* b = VarNode("b")->assert_is_persistable_var();
  // auto* mul = OpNode("mul", op_type_)->assert_node_satisfied(inputs_teller0);
  // auto* mul_out = VarNode("mul_out");
  // auto* add =
  //     OpNode("add", "elementwise_add")->assert_node_satisfied(inputs_teller1);
  // auto* Out = VarNode("Out");
  // if (op_type_ == "matmul") {
  //   mul = OpNode("mul", op_type_)->assert_node_satisfied(input_attr_teller);
  // } else if (op_type_ == "matmul_v2") {
  //   mul = OpNode("mul", op_type_)->assert_node_satisfied(input_attr_teller_v2);
  // }

  // create topology.
  std::vector<PMNode*> CalibNode_input{calibInput};
  std::vector<PMNode*> Conv2dNode_input{calibOutput, convFilter, convBias};
  // mul_inputs >> *mul >> *mul_out;
  CalibNode_input >> *CalibNode >> *calibOutput;

  // Some op specialities.
  calibOutput->AsIntermediate();
  convNode->AsIntermediate();
  CalibNode->AsIntermediate();

  Conv2dNode_input >> *convNode >> *convOutput;
}

void FpgaConvFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
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
  IR_NODE_LINK_TO(matched.at("Filter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("Bias"), new_op_node);
  // IR_NODE_LINK_TO(matched.at("Filter0_scale"), new_op_node);
  // Filter0_scale是一个ATTR，不需要链接到新节点上
  IR_NODE_LINK_TO(new_op_node, matched.at("Output"));
}

cpp::OpDesc FpgaConvFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* conv_info = matched.at("conv2d")->stmt()->op_info();
  auto* calib_info = matched.at("calib")->stmt()->op_info();

  cpp::OpDesc op_desc;
  op_desc.SetType("calib_conv2d");

  // 1. 输入应当来自 calib 的输入，而不是 conv2d 的输入
  // 因为融合前结构一般是：Input -> calib -> conv2d -> Output
  op_desc.SetInput("Input", calib_info->Input("Input"));

  // 2. conv2d 的权重、bias 保持原来的变量名
  op_desc.SetInput("Filter", conv_info->Input("Filter"));

  if (conv_info->HasInput("Bias") && !conv_info->Input("Bias").empty()) {
    op_desc.SetInput("Bias", conv_info->Input("Bias"));
  }

  // 3. 输出使用 conv2d 的最终输出
  op_desc.SetOutput("Output", conv_info->Output("Output"));


  // 如果 calib 的 scale 是 attr，而不是 input，也要保留
  if (calib_info->HasAttr("scale")) {
    op_desc.SetAttr("scale", calib_info->GetAttr<float>("scale"));
  }

  if (calib_info->HasAttr("scale_vct")) {
    op_desc.SetAttr("scale_vct",
                    calib_info->GetAttr<std::vector<float>>("scale_vct"));
  }

  // 5. 保留 conv2d 属性
  if (conv_info->HasAttr("Filter0_scale")) {
    op_desc.SetAttr(
        "Filter0_scale",
        conv_info->GetAttr<std::vector<float>>("Filter0_scale"));
  }

  if (conv_info->HasAttr("Input0_scale")) {
    op_desc.SetAttr(
        "scale_in",
        conv_info->GetAttr<std::vector<float>>("Input0_scale"));
  }

  op_desc.SetAttr("bit_length", conv_info->GetAttr<int>("bit_length"));
  op_desc.SetAttr("dilations", conv_info->GetAttr<std::vector<int>>("dilations"));
  op_desc.SetAttr("groups", conv_info->GetAttr<int>("groups"));
  op_desc.SetAttr("padding_algorithm",
                  conv_info->GetAttr<std::string>("padding_algorithm"));
  op_desc.SetAttr("strides", conv_info->GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings", conv_info->GetAttr<std::vector<int>>("paddings"));

  return op_desc;
}


}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
