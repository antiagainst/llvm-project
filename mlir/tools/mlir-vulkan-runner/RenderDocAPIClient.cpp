//===- RenderDocClient.cpp - RenderDoc API Client -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RenderDocAPIClient.h"

#include <dlfcn.h>
#include <iostream>

using namespace mlir;

static const char kRenderDocLibName[] = "librenderdoc.so";

RenderDocAPIClient::RenderDocAPIClient()
    : renderDocLibrary(nullptr), renderDocAPI(nullptr) {}

RenderDocAPIClient::~RenderDocAPIClient() { disconnect(); }

LogicalResult RenderDocAPIClient::connect() {
  if (renderDocAPI)
    return success();

  renderDocLibrary = dlopen(kRenderDocLibName, RTLD_LAZY);
  if (renderDocLibrary == nullptr) {
    std::cerr << "cannot load RenderDoc library: " << dlerror();
    return failure();
  }

  void *symbol = dlsym(renderDocLibrary, "RENDERDOC_GetAPI");
  assert(symbol && "RENDERDOC_GetAPI must exist in RenderDoc library");
  pRENDERDOC_GetAPI getAPIFn = reinterpret_cast<pRENDERDOC_GetAPI>(symbol);
  int ret = getAPIFn(eRENDERDOC_API_Version_1_4_0, (void **)&renderDocAPI);

  if (ret != 1) {
    renderDocAPI = nullptr;
    dlclose(renderDocLibrary);
    std::cerr << "failed to get RenderDoc API 1.4.0 object";
    return failure();
  }

  return success();
}

void RenderDocAPIClient::disconnect() {
  if (renderDocAPI == nullptr)
    return;

  if (isCapturing())
    endCapture();

  renderDocAPI = nullptr;
  dlclose(renderDocLibrary);
}

void RenderDocAPIClient::startCapture() {
  assert(renderDocAPI && "must connect for capturing");
  assert(!isCapturing() && "capture already in progress");
  renderDocAPI->StartFrameCapture(/*device=*/nullptr, /*window=*/nullptr);
}

void RenderDocAPIClient::endCapture() {
  assert(renderDocAPI && "must connect for capturing");
  assert(isCapturing() && "cannot stop while not capturing");
  renderDocAPI->EndFrameCapture(/*device=*/nullptr, /*window=*/nullptr);
}

bool RenderDocAPIClient::isCapturing() {
  return renderDocAPI && renderDocAPI->IsFrameCapturing() == 1;
}

const char *RenderDocAPIClient::getCapturePath() {
  assert(renderDocAPI && "must connect for capturing");
  return renderDocAPI->GetCaptureFilePathTemplate();
}
