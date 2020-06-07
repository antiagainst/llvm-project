//===- CanonicalizeSPIRVPass.cpp - Canonicalize SPIR-V operations ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs recursive canonicalization on SPIR-V operations in a
// spv.module. It starts with the region of a spv.module and descends into
// nested regions in other SPIR-V ops like spv.func, spv.selection, spv.loop.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {
class CanonicalizeSPIRVPass
    : public SPIRVCanonicalizeBase<CanonicalizeSPIRVPass> {
public:
  void runOnOperation() override;

private:
  void canonicalize(Operation *rootOp, OwningRewritePatternList &patterns);
};
} // namespace

void CanonicalizeSPIRVPass::runOnOperation() {
  spirv::ModuleOp module = getOperation();
  Dialect *spvDialect = module.getDialect();

  OwningRewritePatternList patterns;
  auto *context = &getContext();
  for (auto *op : context->getRegisteredOperations()) {
    if (&op->dialect == spvDialect)
      op->getCanonicalizationPatterns(patterns, context);
  }

  canonicalize(module, patterns);
}

void CanonicalizeSPIRVPass::canonicalize(Operation *rootOp,
                                         OwningRewritePatternList &patterns) {
  for (Region &region : rootOp->getRegions()) {
    for (Operation &op : region.getOps())
      if (isa<spirv::FuncOp>(op) || isa<spirv::SelectOp>(op) ||
          isa<spirv::LoopOp>(op))
        canonicalize(&op, patterns);
  }

  applyPatternsAndFoldGreedily(rootOp->getRegions(), patterns);
}

std::unique_ptr<OperationPass<spirv::ModuleOp>>
mlir::spirv::createCanonicalizeSPIRVPass() {
  return std::make_unique<CanonicalizeSPIRVPass>();
}
