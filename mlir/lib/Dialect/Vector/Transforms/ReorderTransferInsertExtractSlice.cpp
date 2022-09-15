//===- ReorderTransferInsertExtractSlice.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir-vector-reorder-transfer"

using namespace mlir;
using namespace mlir::vector;

/// Returns true if all rank reduced in the given `extractOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::ExtractSliceOp extractOp,
                                        unsigned trailingRank) {
  if (extractOp.getSourceType().getRank() == extractOp.getType().getRank())
    return true;

  RankedTensorType inferredType = extractOp.inferResultType(
      extractOp.getSourceType(), extractOp.getMixedOffsets(),
      extractOp.getMixedSizes(), extractOp.getMixedStrides());
  return extractOp.getType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

/// Returns true if all rank reduced in the given `insertOp` happen in leading
/// dimensions earlier than last `trailingRank` dimensions.
static bool areAllRankReducedLeadingDim(tensor::InsertSliceOp insertOp,
                                        unsigned trailingRank) {
  // If no reduced ranks then simply return true.
  if (insertOp.getSourceType().getRank() == insertOp.getDestType().getRank())
    return true;

  // Infer the small type by extracting from the large type.
  RankedTensorType inferredType = tensor::ExtractSliceOp::inferResultType(
      insertOp.getDestType(), insertOp.getMixedOffsets(),
      insertOp.getMixedSizes(), insertOp.getMixedStrides());
  return insertOp.getSourceType().getShape().take_back(trailingRank) ==
         inferredType.getShape().take_back(trailingRank);
}

namespace {

/// Moves vector.transfer_write ops that are sandwched inside a matching
/// tensor.extract_slice and tensor.insert_slice pair to be after the slice op
/// pair, so that we can cancel the slice op pair later.
///
/// For example, given the following IR:
/// ```
/// %extract = tensor.extract_slice %input[0, 0, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///              : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// %write0 = vector.transfer_write %val0, %extract[%c0, %c0, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %insert = tensor.insert_slice %write1 into %input[0, 0, 0, 0] [1, 1, 2, 4]
///              [1, 1, 1, 1] : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// ```
/// We can fold it into
/// ```mlir
/// %write0 = vector.transfer_write %val0, %input[%c0, %c0, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// %write1 = vector.transfer_write %val1, %write0[%c0, %c1, %c0]
///              {in_bounds = [true]} : vector<4xf32>, tensor<1x2x4xf32>
/// ```
/// In order to avoid causing problems for bufferization, which relies on the
/// extract_slice -> ... > insert_slice structure, this pattern requires all the
/// offsets/sizes/strides to be constants.
struct ReorderTransferWriteAsInsertSource final
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "looking at insertOp: " << insertOp << "\n");
    auto writeOp = insertOp.getSource().getDefiningOp<TransferWriteOp>();
    if (!writeOp)
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "last write op: " << writeOp << "\n");

    Value writeDest = writeOp.getSource();
    // Allow a chain of vector.transfer_write ops that build upon one another.
    // It's common to see that after vector unrolling.
    while (auto prevOp = writeDest.getDefiningOp<TransferWriteOp>())
      writeDest = prevOp.getSource();
    auto extractOp = writeDest.getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp)
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "extract op: " << extractOp << "\n");

    Value insertDest = insertOp.getDest();
    // while (auto prevOp = insertDest.getDefiningOp<TransferWriteOp>())
    //   insertDest = prevOp.getSource();
    if (extractOp.getSource() != insertDest) {
      LLVM_DEBUG(llvm::dbgs() << "mismatched sources\n");
      return failure();
    }

    // Check that the extract and insert slice op has matching offsets, sizes,
    // and strides. This makes sure they can be folded away afterwards.
    const auto isSame = [](OpFoldResult a, OpFoldResult b) { return a == b; };
    if (!extractOp.isSameAs(insertOp, isSame)) {
      LLVM_DEBUG(llvm::dbgs() << "mismatched parameters\n");
      return rewriter.notifyMatchFailure(insertOp, "mismatched parameters");
    }

    // Make sure the transfer_write op has minor identity and all reduced rank
    // are in leading dimensions. This avoid complicated rank reducing issues
    // when swap the transfer and slice op.
    int64_t largeTensorRank = insertOp.getType().getRank();
    int64_t smallTensorRank = insertOp.getSourceType().getRank();
    int64_t vectorRank = writeOp.getVectorType().getRank();
    if (!writeOp.getPermutationMap().isMinorIdentity()) {
      LLVM_DEBUG(llvm::dbgs() << "not minor identity map\n");
      return rewriter.notifyMatchFailure(insertOp, "not minor identity map");
    }
    if (!areAllRankReducedLeadingDim(extractOp, smallTensorRank)) {
      LLVM_DEBUG(llvm::dbgs() << "not leading rank reduced\n");
      return rewriter.notifyMatchFailure(insertOp, "not leading rank reduced");
    }

    // Restrict to constant offsets/sizes/strides for now. This avoids problems
    // with bufferization, where we need the IR structure of extract_slice ->
    // ... -> insert_slice. For that case the parameters are typically loop
    // dependent.
    const unsigned index =
        tensor::InsertSliceOp::getOffsetSizeAndStrideStartOperandIndex();
    for (Value v : insertOp->getOperands().drop_front(index))
      if (!matchPattern(v, m_Constant())) {
        LLVM_DEBUG(llvm::dbgs() << "not cst parameters\n");
        return rewriter.notifyMatchFailure(insertOp, "non-const parameters");
      }

    Location loc = insertOp.getLoc();
    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
        loc, writeOp.getSource(), insertOp.getDest(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    // Prepend zeros to the indices to match the large tensor, if the extract
    // slice op is rank reducing.
    SmallVector<Value> newIndices;
    newIndices.reserve(largeTensorRank);
    int64_t reducedRank = largeTensorRank - smallTensorRank;
    for (int i = 0; i < reducedRank; ++i) {
      OpFoldResult offset = insertOp.getMixedOffsets()[i];
      newIndices.push_back(
          getValueOrCreateConstantIndexOp(rewriter, loc, offset));
    }
    for (int i = 0; i < smallTensorRank; ++i) {
      OpFoldResult offset = insertOp.getMixedOffsets()[i + reducedRank];
      newIndices.push_back(rewriter.create<arith::AddIOp>(
          loc, writeOp.getIndices()[i],
          getValueOrCreateConstantIndexOp(rewriter, loc, offset)));
    }

    auto newMap = AffineMap::getMinorIdentityMap(largeTensorRank, vectorRank,
                                                 writeOp.getContext());

    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        insertOp, writeOp.getVector(), newInsertOp.getResult(), newIndices,
        AffineMapAttr::get(newMap), writeOp.getMask(),
        writeOp.getInBoundsAttr());
    LLVM_DEBUG(llvm::dbgs() << "good now!\n");
    return success();
  }
};

struct ReorderTransferWriteAsInsertDest final
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    auto writeOp = insertOp.getDest().getDefiningOp<TransferWriteOp>();
    if (!writeOp)
      return rewriter.notifyMatchFailure(insertOp, "not inserting into write");
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(insertOp, "not minor identity map");

    if (!insertOp.hasUnitStride())
      return rewriter.notifyMatchFailure(insertOp, "not unit stride");
    if (!areAllRankReducedLeadingDim(insertOp,
                                     insertOp.getSourceType().getRank()))
      return rewriter.notifyMatchFailure(insertOp, "not leading rank reduced");

    unsigned writeTensorRank = writeOp.getSource().getType().getRank();
    unsigned writeReducedRank = writeOp.getLeadingShapedRank();

    SmallVector<OpFoldResult> writeOffsets;
    writeOffsets.reserve(writeTensorRank);
    llvm::append_range(writeOffsets, writeOp.getIndices());

    SmallVector<OpFoldResult> writeSizes;
    writeSizes.reserve(writeTensorRank);
    for (unsigned i = 0; i < writeReducedRank; ++i)
      writeSizes.push_back(rewriter.getIndexAttr(1));
    for (unsigned i = writeReducedRank; i < writeTensorRank; ++i)
      writeSizes.push_back(rewriter.getIndexAttr(
          writeOp.getVectorType().getDimSize(i - writeReducedRank)));

    SmallVector<OpFoldResult> insertOffsets = insertOp.getMixedOffsets();
    SmallVector<OpFoldResult> insertSizes = insertOp.getMixedSizes();

    if (!areDisjointRanges(writeOffsets, writeSizes, insertOffsets,
                           insertSizes))
      return rewriter.notifyMatchFailure(insertOp, "not disjoint ranges");

    auto newInsertOp = rewriter.create<tensor::InsertSliceOp>(
        insertOp.getLoc(), insertOp.getSource(), writeOp.getSource(),
        insertOp.getMixedOffsets(), insertOp.getMixedSizes(),
        insertOp.getMixedStrides());

    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        insertOp, writeOp.getVector(), newInsertOp.getResult(),
        writeOp.getIndices(), writeOp.getPermutationMapAttr(),
        writeOp.getMask(), writeOp.getInBoundsAttr());

    return success();
  }
};

} // namespace

void vector::populateVectorReorderTransferExtractInsertSlicePatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ReorderTransferWriteAsInsertSource,
               ReorderTransferWriteAsInsertDest>(patterns.getContext(),
                                                 benefit);
}
