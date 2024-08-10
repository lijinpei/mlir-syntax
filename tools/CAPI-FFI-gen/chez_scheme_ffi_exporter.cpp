#include "capi_ffi_generator.h"

#include "clang/AST/Decl.h"

using namespace llvm;
using namespace clang;

namespace capi_ffi_gen {
namespace {
struct ChezSchemeFFIExporter : FFIExporter {
  const YAML::Node &config;
  ChezSchemeFFIExporter(llvm::StringRef outputPath, const YAML::Node &config,
                        FFIExporterContext context)
      : config(config), FFIExporter("chez-scheme", context) {}
  bool exportTypeDefNameDecl(clang::TypedefNameDecl *) override { return true; }
  bool exportEnumDecl(EnumDecl *enumDecl) override { return true; }
  bool exportRecordDecl(RecordDecl *recordDecl) override { return true; }
  bool exportFunctionDecl(clang::FunctionDecl *) override { return true; }
  void finishTranslationUnit(ASTContext &Ctx) override {}
  bool handleMainFileInclude(llvm::StringRef fileName) override { return true; }
};
} // namespace
std::unique_ptr<FFIExporter>
createChezSchemeExporter(llvm::StringRef outputPath, const YAML::Node &config,
                         FFIExporterContext context) {
  return std::make_unique<ChezSchemeFFIExporter>(outputPath, config, context);
}
} // namespace capi_ffi_gen
