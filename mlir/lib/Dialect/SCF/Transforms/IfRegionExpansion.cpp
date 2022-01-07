//===- IfRegionExpansion.cpp - Pull ops into scf.if Region ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns and passes for expanding scf.if's regions
// by pulling in ops before and after the scf.if op into both regions of the
// scf.if op.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-scf-expand-if-region"

using namespace mlir;

static constexpr char kExpandedIfMarker[] = "__expanded_if_regions__";

/// Pulls ops at the same nest level as the given `ifOp` into both regions of
/// the if `ifOp`.
static FailureOr<scf::IfOp> pullOpsIntoIfRegions(scf::IfOp ifOp,
                                                 RewriterBase &rewriter) {
  // Need to pull ops into both regions.
  if (!ifOp.elseBlock())
    return failure();

  // Expect to only have one block in the enclosing region. This is the common
  // case for the level where we have structured control flows and it avoids
  // traditional control flow and simplifies the analysis.
  if (!llvm::hasSingleElement(ifOp->getParentRegion()->getBlocks()))
    return failure();

  SmallVector<Operation *> allOps;
  for (Operation &op : ifOp->getBlock()->without_terminator())
    allOps.push_back(&op);

  // If no ops before or after the if op, there is nothing to do.
  if (allOps.size() == 1)
    return failure();

  auto prevOps = llvm::makeArrayRef(allOps).take_while(
      [&ifOp](Operation *op) { return op != ifOp.getOperation(); });
  auto nextOps = llvm::makeArrayRef(allOps).drop_front(prevOps.size() + 1);

  // Require all previous ops to have on side effects, so that after cloning
  // them into both regions, we can rely on DCE to remove them.
  if (llvm::any_of(prevOps, [](Operation *op) {
        return !MemoryEffectOpInterface::hasNoEffect(op);
      }))
    return failure();

  Operation *parentTerminator = ifOp->getBlock()->getTerminator();
  TypeRange resultTypes = ifOp.getResultTypes();
  if (!nextOps.empty()) {
    // The if op should yield the values used by the terminator.
    resultTypes = parentTerminator->getOperandTypes();
  }

  auto newIfOp = rewriter.create<scf::IfOp>(
      ifOp.getLoc(), resultTypes, ifOp.getCondition(), ifOp.elseBlock());

  auto pullIntoBlock = [&](Block *newblock, Block *oldBlock) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(newblock);
    BlockAndValueMapping bvm;

    // Clone all ops defined before the original if op.
    for (Operation *prevOp : prevOps)
      rewriter.clone(*prevOp, bvm);

    // Clone all ops defined inside the original if block.
    for (Operation &blockOp : oldBlock->without_terminator())
      rewriter.clone(blockOp, bvm);

    if (nextOps.empty()) {
      // If the if op needs to return value, its builder won't automatically
      // insert terminators. Just clone the old one here.
      if (newIfOp->getNumResults())
        rewriter.clone(*oldBlock->getTerminator(), bvm);
      return;
    }

    // There are ops after the old if op. Uses of the old if op should be
    // replaced by the cloned yield value.
    auto oldYieldOp = cast<scf::YieldOp>(oldBlock->back());
    for (int i = 0, e = ifOp->getNumResults(); i < e; ++i) {
      bvm.map(ifOp->getResult(i), bvm.lookup(oldYieldOp.getOperand(i)));
    }

    // Clone all ops defined after the original if op. While doing that, we need
    // to check whether the op is used by the terminator. If so, we need to
    // yield its result value at the proper index.
    SmallVector<Value> yieldValues(newIfOp.getNumResults());
    for (Operation *nextOp : nextOps) {
      rewriter.clone(*nextOp, bvm);
      for (OpOperand &use : nextOp->getUses()) {
        if (use.getOwner() == parentTerminator) {
          unsigned index = use.getOperandNumber();
          yieldValues[index] = bvm.lookup(use.get());
        }
      }
    }

    if (!yieldValues.empty()) {
      // Again the if builder won't insert terminators automatically.
      rewriter.create<scf::YieldOp>(ifOp.getLoc(), yieldValues);
    }
  };

  pullIntoBlock(newIfOp.thenBlock(), ifOp.thenBlock());
  pullIntoBlock(newIfOp.elseBlock(), ifOp.elseBlock());

  if (nextOps.empty()) {
    rewriter.replaceOp(ifOp, newIfOp->getResults());
  } else {
    // Update the terminator to use the new if op's results.
    rewriter.updateRootInPlace(parentTerminator, [&]() {
      parentTerminator->setOperands(newIfOp->getResults());
    });
    // We have pulled in all ops following the if op into both regions. Now
    // remove them all. Do this in the reverse order.
    for (Operation *op : llvm::reverse(nextOps))
      rewriter.eraseOp(op);
    rewriter.eraseOp(ifOp);
  }

  return newIfOp;
}

namespace {

struct IfRegionExpansionPattern final : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp->hasAttr(kExpandedIfMarker))
      return failure();

    auto newOp = pullOpsIntoIfRegions(ifOp, rewriter);
    if (failed(newOp))
      return failure();

    newOp.getValue()->setAttr(kExpandedIfMarker, rewriter.getUnitAttr());
    return success();
  }
};

struct IfRegionExpansion : public SCFIfRegionExpansionBase<IfRegionExpansion> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<IfRegionExpansionPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

} // namespace

void scf::populateIfRegionExpansionPatterns(RewritePatternSet &patterns) {
  patterns.insert<IfRegionExpansionPattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::createIfRegionExpansionPass() {
  return std::make_unique<IfRegionExpansion>();
}
