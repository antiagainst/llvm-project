// RUN: mlir-opt %s -linalg-generalize-named-ops -split-input-file | FileCheck %s

func @generalize_conv(%input : memref<1x225x225x3xf32>, %filter: memref<3x3x3x32xf32>, %output: memref<1x112x112x32xf32>) {
  linalg.conv(%filter, %input, %output) {dilations = [2, 3], strides = [4, 5]} : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>, memref<1x112x112x32xf32>
  return
}

// CHECK: #[[FILTER_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
// CHECK:  #[[INPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 4 + d3 * 2, d2 * 5 + d4 * 3, d5)>
// CHECK: #[[OUTPUT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>

// CHECK: func @generalize_conv
// CHECK-SAME:  %[[INPUT:.+]]: memref<1x225x225x3xf32>
// CHECK-SAME: %[[FILTER:.+]]: memref<3x3x3x32xf32>
// CHECK-SAME: %[[OUTPUT:.+]]: memref<1x112x112x32xf32>

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[FILTER_MAP]], #[[INPUT_MAP]], #[[OUTPUT_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "window", "window", "reduction", "parallel"]
// CHECK-SAME:  ins(%[[FILTER]], %[[INPUT]]
// CHECK-SAME: outs(%[[OUTPUT]]

// CHECK: ^{{.*}}(%[[FILTER_ARG:.+]]: f32, %[[INPUT_ARG:.+]]: f32, %[[OUTPUT_ARG:.+]]: f32)
// CHECK:   %[[MUL:.+]] = mulf %[[FILTER_ARG]], %[[INPUT_ARG]]
// CHECK:   %[[ADD:.+]] = addf %[[MUL]], %[[OUTPUT_ARG]]
// CHECK:   linalg.yield %[[ADD]]
