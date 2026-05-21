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

#include "lite/core/optimizer/mir/subgraph/fpga_conv_fuse_pass_with_branch.h"
#include <list>
#include <memory>
#include <vector>
#include "lite/core/optimizer/mir/subgraph/fpga_conv_fuser.h"
#include "lite/core/optimizer/mir/subgraph/fpga_conv_fuser_with_branch.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void FpgaConvFusePassWithBranch::Apply(const std::unique_ptr<SSAGraph>& graph) {
      std::cout << "apply fpga conv fuse pass with branch" << std::endl;
      // fusion::FpgaConvFuser fuser("", "");
      // fuser(graph.get());
      fusion::FpgaConvFuserWithBranch fuser1("", "");
      fuser1(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(fpga_conv_fuse_pass_with_branch, paddle::lite::mir::FpgaConvFusePassWithBranch)
    .BindTargets({TARGET(kAny)});
    // .BindKernel("calib_conv2d");
