#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Allocator.h"

#include "fmt/format.h"
#include "yaml-cpp/yaml.h"

#include <memory>

namespace clang {
class EnumDecl;
class RecordDecl;
class ASTContext;
class DiagnosticsEngine;
class TypedefNameDecl;
class FunctionDecl;
class ASTContext;
} // namespace clang

namespace capi_ffi_gen {

class ExportCAPIConsumer;

struct FFIExporterContext {
  void *impl;
};

struct FFIExporter {
  llvm::StringRef lang;
  bool hasError = false;
  unsigned indentLevel = 0;
  FFIExporterContext context;
  FFIExporter(FFIExporterContext context) : context(context) {}
  FFIExporter(llvm::StringRef lang, FFIExporterContext context)
      : lang(lang), context(context) {}
  void diagError(llvm::Twine message);
  ExportCAPIConsumer &getConsumer();
  clang::ASTContext &getASTContext();
  clang::DiagnosticsEngine &getDiagnostics();
  static constexpr unsigned NumIndent = 2;
  virtual ~FFIExporter();
  virtual bool exportTypeDefNameDecl(clang::TypedefNameDecl *) = 0;
  virtual bool exportEnumDecl(clang::EnumDecl *) = 0;
  virtual bool exportRecordDecl(clang::RecordDecl *) = 0;
  virtual bool exportFunctionDecl(clang::FunctionDecl *) = 0;
  virtual void finishTranslationUnit(clang::ASTContext &Ctx) = 0;
  virtual bool handleMainFileInclude(llvm::StringRef fileName) = 0;
};

struct IncIndent {
  FFIExporter &exporter;
  IncIndent(FFIExporter &exporter) : exporter(exporter) {
    exporter.indentLevel += 1;
  }
  ~IncIndent() { exporter.indentLevel -= 1; }
};

llvm::StringRef interpolate_config_string(llvm::StringRef str,
                                          llvm::BumpPtrAllocator &allocator);

std::unique_ptr<FFIExporter> createRustExporter(llvm::StringRef outputPath,
                                                const YAML::Node &config,
                                                FFIExporterContext context);
std::unique_ptr<FFIExporter>
createChezSchemeExporter(llvm::StringRef outputPath, const YAML::Node &config,
                         FFIExporterContext context);
} // namespace capi_ffi_gen
