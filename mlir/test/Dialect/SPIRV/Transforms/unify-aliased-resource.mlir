// RUN: mlir-opt -split-input-file -spirv-unify-aliased-resource -verify-diagnostics %s | FileCheck %s

spirv.module Logical GLSL450 {
  spirv.GlobalVariable @var00_v4f16 bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf16>, stride=8> [0])>, StorageBuffer>
  spirv.GlobalVariable @var00_f32   bind(0, 0) {aliased} : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>

  spirv.func @scalar_type_bitwidth_larger_than_vector(%i0: i32) -> f32 "None" {
    %c0 = spirv.Constant 0 : i32

    %addr = spirv.mlir.addressof @var00_f32 : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
    %ac = spirv.AccessChain %addr[%c0, %i0] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32
    %val = spirv.Load "StorageBuffer" %ac : f32

    spirv.ReturnValue %val : f32
  }
}

// CHECK-LABEL: spirv.module

// CHECK-NOT: @var00_f32
//     CHECK: spirv.GlobalVariable @var00_v4f16_int bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xi16>, stride=8> [0])>, StorageBuffer>
// CHECK-NOT: @var00_f32

//     CHECK: spirv.func @scalar_type_bitwidth_larger_than_vector(%[[IDX:.+]]: i32)
//     CHECK:   %[[C0:.+]] = spirv.Constant 0 : i32
//     CHECK:   %[[ADDR:.+]] = spirv.mlir.addressof @var00_v4f16_int
//     CHECK:   %[[C2:.+]] = spirv.Constant 2 : i32
//     CHECK:   %[[DIV:.+]] = spirv.SDiv %[[IDX]], %[[C2]] : i32
//     CHECK:   %[[MOD:.+]] = spirv.SMod %[[IDX]], %[[C2]] : i32
//     CHECK:   %[[AC0:.+]] = spirv.AccessChain %[[ADDR]][%[[C0]], %[[DIV]], %[[MOD]]]
//     CHECK:   %[[LD0:.+]] = spirv.Load "StorageBuffer" %[[AC0]] : i16
//     CHECK:   %[[C1:.+]] = spirv.Constant 1 : i32
//     CHECK:   %[[ADD:.+]] = spirv.IAdd %[[MOD]], %[[C1]] : i32
//     CHECK:   %[[AC1:.+]] = spirv.AccessChain %[[ADDR]][%[[C0]], %[[DIV]], %[[ADD]]]
//     CHECK:   %[[LD1:.+]] = spirv.Load "StorageBuffer" %[[AC1]] : i16
//     CHECK:   %[[CC:.+]] = spirv.CompositeConstruct %[[LD0]], %[[LD1]]
//     CHECK:   %[[BC:.+]] = spirv.Bitcast %[[CC]] : vector<2xi16> to f32
//     CHECK:   spirv.ReturnValue %[[BC]]
