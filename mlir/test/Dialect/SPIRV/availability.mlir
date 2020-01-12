// RUN: mlir-opt -disable-pass-threading -test-spirv-op-availability %s | FileCheck %s

// CHECK-LABEL: iadd
func @iadd(%arg: i32) -> i32 {
  // CHECK: has no min version requirement
  // CHECK: has no max version requirement
  // CHECK: has no extension requirement
  // CHECK: has no capability requirement
  %0 = spv.IAdd %arg, %arg: i32
  return %0: i32
}

// CHECK: atomic_compare_exchange_weak
func @atomic_compare_exchange_weak(%ptr: !spv.ptr<i32, Workgroup>, %value: i32, %comparator: i32) -> i32 {
  // CHECK: min version: V_1_0
  // CHECK: max version: V_1_3
  // CHECK: extensions: [ ]
  // CHECK: capabilities: [ [Kernel] ]
  %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Release" "Acquire" %ptr, %value, %comparator: !spv.ptr<i32, Workgroup>
  return %0: i32
}

// CHECK-LABEL: subgroup_ballot
func @subgroup_ballot(%predicate: i1) -> vector<4xi32> {
  // CHECK: min version: V_1_3
  // CHECK: has no max version requirement
  // CHECK: has no extension requirement
  // CHECK: capabilities: [ [GroupNonUniformBallot] ]
  %0 = spv.GroupNonUniformBallot "Workgroup" %predicate : vector<4xi32>
  return %0: vector<4xi32>
}

// CHECK-LABEL: module_logical_glsl450
func @module_logical_glsl450() {
  // CHECK: has no min version requirement
  // CHECK: has no max version requirement
  // CHECK: has no extension requirement
  // CHECK: spv.module capabilities: [ [Shader] ]
  spv.module "Logical" "GLSL450" { }
  return
}

// CHECK-LABEL: module_physical_storage_buffer64_vulkan
func @module_physical_storage_buffer64_vulkan() {
  // CHECK: spv.module has no min version requirement
  // CHECK: spv.module has no max version requirement
  // CHECK: spv.module extensions: [ [SPV_EXT_physical_storage_buffer, SPV_KHR_physical_storage_buffer] [SPV_KHR_vulkan_memory_model] ]
  // CHECK: spv.module capabilities: [ [PhysicalStorageBufferAddresses] [VulkanMemoryModel] ]
  spv.module "PhysicalStorageBuffer64" "Vulkan" { }
  return
}
