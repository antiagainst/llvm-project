//===- StringExtras.cpp - String utilities used by MLIR -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/StringExtras.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

std::string mlir::convertToSnakeCase(StringRef input) {
  std::string snakeCase;
  snakeCase.reserve(input.size());
  for (auto c : input) {
    if (std::isupper(c)) {
      if (!snakeCase.empty() && snakeCase.back() != '_') {
        snakeCase.push_back('_');
      }
      snakeCase.push_back(llvm::toLower(c));
    } else {
      snakeCase.push_back(c);
    }
  }
  return snakeCase;
}

std::string mlir::convertToCamelCase(StringRef input, bool capitalizeFirst) {
  if (input.empty())
    return "";

  std::string output;
  output.reserve(input.size());
  size_t pos = 0;
  if (capitalizeFirst && std::islower(input[pos])) {
    output.push_back(llvm::toUpper(input[pos]));
    pos++;
  }
  while (pos < input.size()) {
    auto cur = input[pos];
    if (cur == '_') {
      if (pos && (pos + 1 < input.size())) {
        if (std::islower(input[pos + 1])) {
          output.push_back(llvm::toUpper(input[pos + 1]));
          pos += 2;
          continue;
        }
      }
    }
    output.push_back(cur);
    pos++;
  }
  return output;
}
