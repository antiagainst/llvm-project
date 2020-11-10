//===- Generalization.cpp - linalg named ops to generic ops  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg generalization pass. It converts named
// Linalg ops to linalg.generic ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;

namespace {

/// Base class for all linalg generalization patterns. A subclass must provide
/// the following method:
///   linalg::GenericOp createGenericOp(RootOp, PatternRewriter &)
/// for creating the generic op.
template <typename ConcretePattern, typename RootOp>
struct LinalgGeneralizationPattern : OpRewritePattern<RootOp> {
  LinalgGeneralizationPattern(MLIRContext *context, linalg::LinalgMarker marker,
                              PatternBenefit benefit = 1)
      : OpRewritePattern<RootOp>(context, benefit), marker(std::move(marker)) {}

  LogicalResult matchAndRewrite(RootOp rootOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(rootOp.getOperation());
    if (!linalgOp)
      return failure();
    if (failed(marker.checkAndNotify(rewriter, linalgOp)))
      return failure();

    auto *pattern = static_cast<const ConcretePattern *>(this);
    linalg::GenericOp genericOp = pattern->createGenericOp(rootOp, rewriter);
    if (!genericOp)
      return failure();

    rewriter.replaceOp(rootOp, genericOp.getResults());
    marker.replaceLinalgMarker(rewriter, genericOp.getOperation());
    return success();
  }

private:
  linalg::LinalgMarker marker;
};

struct GeneralizeConvOp
    : public LinalgGeneralizationPattern<GeneralizeConvOp, linalg::ConvOp> {
  using LinalgGeneralizationPattern::LinalgGeneralizationPattern;

  linalg::GenericOp createGenericOp(linalg::ConvOp,
                                    PatternRewriter &rewriter) const;
};

struct LinalgGeneralizationPass
    : public LinalgGeneralizationBase<LinalgGeneralizationPass> {
  void runOnFunction() override;
};

} // namespace

linalg::GenericOp
GeneralizeConvOp::createGenericOp(linalg::ConvOp convOp,
                                  PatternRewriter &rewriter) const {
  SmallVector<AffineMap, 4> indexingMaps = convOp.getIndexingMaps();
  auto iterators =
      llvm::to_vector<4>(convOp.iterator_types().getAsValueRange<StringAttr>());
  return rewriter.create<linalg::GenericOp>(
      convOp.getLoc(), /*resultTensorTypes=*/ArrayRef<Type>(),
      convOp.getInputBuffers(), convOp.getOutputBuffers(),
      /*initTensors=*/ValueRange(), indexingMaps, iterators,
      [](OpBuilder &bodyBuilder, Location bodyLoc, ValueRange bodyArgs) {
        Value mul =
            bodyBuilder.create<MulFOp>(bodyLoc, bodyArgs[0], bodyArgs[1]);
        Value add = bodyBuilder.create<AddFOp>(bodyLoc, mul, bodyArgs[2]);
        bodyBuilder.create<linalg::YieldOp>(bodyLoc, add);
      });
}

void LinalgGeneralizationPass::runOnFunction() {
  FuncOp func = getFunction();
  OwningRewritePatternList patterns;
  linalg::populateLinalgConvGeneralizationPatterns(&getContext(), patterns);
  applyPatternsAndFoldGreedily(func.getBody(), std::move(patterns));
}

void mlir::linalg::populateLinalgConvGeneralizationPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    linalg::LinalgMarker marker) {
  patterns.insert<GeneralizeConvOp>(context, marker);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgGeneralizationPass() {
  return std::make_unique<LinalgGeneralizationPass>();
}
