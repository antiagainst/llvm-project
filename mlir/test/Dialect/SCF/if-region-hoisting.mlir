// RUN: mlir-opt -split-input-file -if-region-hoisting %s | FileCheck %s

// CHECK-LABEL: func @nothing_to_hoist
func @nothing_to_hoist(%cond: i1, %val: i32) -> i32 {
  %if = scf.if %cond -> i32 {
    scf.yield %val: i32
  } else {
    scf.yield %val: i32
  }
  return %if: i32
}

//      CHECK: scf.if
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: else
// CHECK-NEXT:   scf.yield

// -----

// CHECK-LABEL: func @all_use_from_above
func @all_use_from_above(%cond: i1, %val1: i32, %val2: i32) -> i32 {
  %if = scf.if %cond -> i32 {
    %add = arith.addi %val1, %val2 : i32
    scf.yield %add : i32
  } else {
    %sub = arith.subi %val1, %val2 : i32
    scf.yield %sub : i32
  }
  return %if: i32
}

//      CHECK: %[[ADD:.+]] = arith.addi
// CHECK-NEXT: %[[SUB:.+]] = arith.subi
// CHECK-NEXT: scf.if
// CHECK-NEXT:  scf.yield %[[ADD]]
// CHECK-NEXT: else
// CHECK-NEXT:  scf.yield %[[SUB]]

// -----

// CHECK-LABEL: func @side_effecting_ops
func @side_effecting_ops(%cond: i1, %val: i32, %buffer: memref<3xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.if %cond {
    memref.store %val, %buffer[%c0] : memref<3xi32>
  } else {
    memref.store %val, %buffer[%c1] : memref<3xi32>
  }
  return
}

//      CHECK: scf.if
// CHECK-NEXT:   memref.store
// CHECK-NEXT: else
// CHECK-NEXT:   memref.store

// -----

// CHECK-LABEL: func @interleaving_ops
func @interleaving_ops(%cond: i1, %i1: index, %i2: index, %buffer: memref<?xf32>) {
  scf.if %cond {
    %add = arith.addi %i1, %i2 : index
    %val = memref.load %buffer[%add] : memref<?xf32>
    %sub = arith.subi %i1, %i2 : index
    memref.store %val, %buffer[%sub] : memref<?xf32>
  }
  return
}

//      CHECK: arith.addi
// CHECK-NEXT: arith.subi
// CHECK-NEXT: scf.if

// -----

// CHECK-LABEL: func @dependent_on_side_effecting_ops
func @dependent_on_side_effecting_ops(%cond: i1, %i1: index, %i2: index, %buffer: memref<?xindex>) {
  scf.if %cond {
    %add = arith.addi %i1, %i2 : index
    %val = memref.load %buffer[%add] : memref<?xindex>
    %sub = arith.subi %i1, %val : index
    memref.store %val, %buffer[%sub] : memref<?xindex>
  }
  return
}

//      CHECK: arith.addi
// CHECK-NEXT: scf.if
// CHECK-NEXT:   memref.load
// CHECK-NEXT:   arith.subi
// CHECK-NEXT:   memref.store

// -----

// CHECK-LABEL: func @chain_of_hoisting_ops
//  CHECK-SAME: %[[I1:.+]]: index, %[[I2:.+]]: index
func @chain_of_hoisting_ops(%i1: index, %i2: index, %cond: i1, %buffer: memref<?xf32>) {
  scf.if %cond {
    %add = arith.addi %i1, %i2 : index
    %mul = arith.muli %add, %i2 : index
    %div = arith.divui %mul, %i1 : index
    %val = memref.load %buffer[%div] : memref<?xf32>
    %sub = arith.subi %i1, %i2 : index
    memref.store %val, %buffer[%sub] : memref<?xf32>
  }

  return
}

//      CHECK: %[[ADD:.+]] = arith.addi %[[I1]], %[[I2]]
// CHECK-NEXT: %[[MUL:.+]] = arith.muli %[[ADD]], %[[I2]]
// CHECK-NEXT: %[[DIV:.+]] = arith.divui %[[MUL]], %[[I1]]
// CHECK-NEXT: %[[SUB:.+]] = arith.subi %[[I1]], %[[I2]]
// CHECK-NEXT: scf.if
// CHECK-NEXT:   memref.load %{{.+}}[%[[DIV]]]
// CHECK-NEXT:   memref.store %{{.+}}, %{{.+}}[%[[SUB]]]
