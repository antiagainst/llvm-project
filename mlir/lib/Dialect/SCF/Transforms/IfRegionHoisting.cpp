//===- IfRegionHoisting.cpp - Hoist ops out of scf.if region --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns and passes to hoist non-side-effecting ops out
// of scf.if's regions.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-scf-hoist-if-region"

using namespace mlir;

namespace {

class IfRegionHoistingPattern final : public OpRewritePattern<scf::IfOp> {
public:
  IfRegionHoistingPattern(MLIRContext *context,
                          llvm::function_ref<bool(Operation *)> hoistControlFn,
                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), shouldHoist(hoistControlFn) {}

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;

    changed |= hoistIfRegion(ifOp, ifOp.getThenRegion(), *ifOp.thenBlock());
    if (ifOp.elseBlock())
      changed |= hoistIfRegion(ifOp, ifOp.getElseRegion(), *ifOp.elseBlock());

    return success(changed);
  }

private:
  bool hoistIfRegion(scf::IfOp ifOp, Region &ifRegion, Block &ifBlock) const {
    // We use two collections here as we need to preserve the order for
    // insertion and this is easiest.
    SmallPtrSet<Operation *, 8> willBeMovedSet;
    SmallVector<Operation *, 8> opsToMove;

    llvm::SetVector<Value> outsideValues;
    getUsedValuesDefinedAbove(ifRegion, outsideValues);

    // Return true if the given value is originally defined outside of the
    // if region or will be moved outside.
    auto isDefinedOutside = [&](Value value) {
      if (outsideValues.contains(value))
        return true;

      auto *definingOp = value.getDefiningOp();
      return definingOp && willBeMovedSet.count(definingOp);
    };

    for (Operation &op : ifBlock.without_terminator()) {
      if (canBeHoistedOutOfRegion(&op, isDefinedOutside) && shouldHoist(&op)) {
        opsToMove.push_back(&op);
        willBeMovedSet.insert(&op);
      }
    }

    for (Operation *op : opsToMove)
      op->moveBefore(ifOp);

    return !willBeMovedSet.empty();
  };

  std::function<bool(Operation *)> shouldHoist;
};

struct IfRegionHoisting final
    : public SCFIfRegionHoistingBase<IfRegionHoisting> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    auto allowAll = [](Operation *) { return true; };
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<IfRegionHoistingPattern>(ctx, allowAll);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void scf::populateIfRegionHoistingPatterns(
    RewritePatternSet &patterns,
    llvm::function_ref<bool(Operation *)> hoistControlFn) {
  patterns.insert<IfRegionHoistingPattern>(patterns.getContext(),
                                           hoistControlFn);
}

std::unique_ptr<Pass> mlir::createIfRegionHoistingPass() {
  return std::make_unique<IfRegionHoisting>();
}
