// RUN: mlir-opt -split-input-file -test-linalg-transform-patterns=test-concretize-padtensor-result-shape -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @only_high_pad
func @only_high_pad(%tensor: tensor<1x224x224x3xf32>, %arg0: index, %arg1: index) {
  %cst = arith.constant 0.0 : f32
  %0 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg0)
  %1 = affine.min affine_map<(d0) -> (d0 * 2 + 3, 224)>(%arg0)
  %2 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%1, %arg0)
  %3 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 3)>(%1, %arg0)
  %4 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg1)
  %5 = affine.min affine_map<(d0) -> (d0 * 2 + 9, 224)>(%arg1)
  %6 = affine.apply affine_map<(d0, d1) -> (d0 - d1 * 2)>(%5, %arg1)
  %7 = affine.apply affine_map<(d0, d1) -> (-d0 + d1 * 2 + 9)>(%5, %arg1)
  %8 = tensor.extract_slice %tensor[0, %0, %4, 0][1, %2, %6, 3][1, 1, 1, 1] : tensor<1x224x224x3xf32> to tensor<1x?x?x3xf32>
  // CHECK: linalg.pad_tensor
  %pad = linalg.pad_tensor %8 low[0, 0, 0, 0] high[0, %3, %7, 0]  {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
  // CHECK: linalg.yield
    linalg.yield %cst : f32
  // CHECK-NEXT: tensor<1x?x?x3xf32> to tensor<1x3x9x3xf32>
  } : tensor<1x?x?x3xf32> to tensor<1x?x?x3xf32>
  "dialect.use"(%pad) : (tensor<1x?x?x3xf32>) -> ()
}

// -----

// CHECK-LABEL: func @both_low_and_high_pad
func @both_low_and_high_pad(%tensor: tensor<1x56x56x144xf32>, %arg0: index, %arg1: index, %arg2: index) {
  %cst = arith.constant 0.0 : f32
  %0 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg0)
  %1 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg0)
  %2 = affine.min affine_map<(d0) -> (d0, 56)>(%1)
  %3 = affine.max affine_map<(d0) -> (d0 + 3, 0)>(%arg0)
  %4 = affine.min affine_map<(d0) -> (d0, 56)>(%3)
  %5 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%4, %2)
  %6 = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 4)>(%0, %4, %2)
  %7 = affine.max affine_map<(d0) -> (0, -d0 + 1)>(%arg1)
  %8 = affine.max affine_map<(d0) -> (d0 - 1, 0)>(%arg1)
  %9 = affine.min affine_map<(d0) -> (d0, 56)>(%8)
  %10 = affine.max affine_map<(d0) -> (d0 + 3, 0)>(%arg1)
  %11 = affine.min affine_map<(d0) -> (d0, 56)>(%10)
  %12 = affine.apply affine_map<(d0, d1) -> (d0 - d1)>(%11, %9)
  %13 = affine.apply affine_map<(d0, d1, d2) -> (-d0 - d1 + d2 + 4)>(%7, %11, %9)
  %14 = tensor.extract_slice %tensor[0, %2, %9, %arg2][1, %5, %12, 16][1, 1, 1, 1] : tensor<1x56x56x144xf32> to tensor<1x?x?x16xf32>
  // CHECK: linalg.pad_tensor
  %pad = linalg.pad_tensor %14 low[0, %0, %7, 0] high[0, %6, %13, 0]  {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
  // CHECK: linalg.yield
    linalg.yield %cst : f32
  // CHECK-NEXT: tensor<1x?x?x16xf32> to tensor<1x4x4x16xf32>
  } : tensor<1x?x?x16xf32> to tensor<1x?x?x16xf32>
  "dialect.use"(%pad) : (tensor<1x?x?x16xf32>) -> ()
}
