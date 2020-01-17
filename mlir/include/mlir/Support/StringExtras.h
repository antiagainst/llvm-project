//===- StringExtras.h - String utilities used by MLIR -----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains string utility functions used within MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STRINGEXTRAS_H
#define MLIR_SUPPORT_STRINGEXTRAS_H

#include "llvm/ADT/StringExtras.h"

#include <cctype>

namespace mlir {
/// Converts a string to snake-case from camel-case by replacing all uppercase
/// letters with '_' followed by the letter in lowercase, except if the
/// uppercase letter is the first character of the string.
std::string convertToSnakeCase(llvm::StringRef input);

/// Converts a string from camel-case to snake_case by replacing all occurrences
/// of '_' followed by a lowercase letter with the letter in
/// uppercase. Optionally allow capitalization of the first letter (if it is a
/// lowercase letter)
std::string convertToCamelCase(llvm::StringRef input,
                               bool capitalizeFirst = false);
} // namespace mlir

#endif // MLIR_SUPPORT_STRINGEXTRAS_H
