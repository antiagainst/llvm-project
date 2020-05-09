//===- RenderDocClient.h - RenderDoc API Client -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a client class for wrapping RenderDoc API. RenderDoc
// is a widely used tool for capturing and analyzing Vulkan applications.
// For more details see https://renderdoc.org/.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRVULKANRUNNER_RENDERDOCAPICLIENT_H_
#define MLIR_TOOLS_MLIRVULKANRUNNER_RENDERDOCAPICLIENT_H_

#include "mlir/Support/LogicalResult.h"
#include <renderdoc.h>

namespace mlir {

/// A client class for wrapping RenderDoc API.
class RenderDocAPIClient {
public:
  RenderDocAPIClient();
  ~RenderDocAPIClient();

  /// Connects to RenderDoc shared library and gets API handles.
  /// This must be called *before* creating a VkInstance.
  LogicalResult connect();

  /// Disconnects from RenderDoc API handles.
  void disconnect();

  /// Starts capturing Vulkan commands.
  /// This must be called *after* creating a VkDevice.
  void startCapture();

  /// Ends capturing Vulkan commands.
  void endCapture();

  /// Returns true if capture is in progress.
  bool isCapturing();

  /// Returns the path for storing captures.
  const char *getCapturePath();

private:
  void *renderDocLibrary;
  RENDERDOC_API_1_4_0 *renderDocAPI;
};

} // namespace mlir

#endif // MLIR_TOOLS_MLIRVULKANRUNNER_RENDERDOCAPICLIENT_H_
