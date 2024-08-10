#include "capi_ffi_generator.h"

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include "fmt/args.h"
#include "yaml-cpp/yaml.h"

#include <string_view>

using namespace clang;
using namespace llvm;

extern "C" char **environ;

namespace capi_ffi_gen {
using EnvArgStore = fmt::dynamic_format_arg_store<fmt::format_context>;
namespace {
EnvArgStore create_interpolate_arg_store(BumpPtrAllocator &allocator) {
  EnvArgStore store;
  static const char EnvVarPrefix[] = "CAPI_EXPORTER";
  const auto EnvVarPrefixLen = sizeof(EnvVarPrefix) - 1;
  for (int i = 0; environ[i]; ++i) {
    const char *ptr = environ[i];
    if (strncmp(ptr, EnvVarPrefix, EnvVarPrefixLen)) {
      continue;
    }
    StringRef env(ptr);
    auto equalPos = env.find('=', EnvVarPrefixLen);
    if (equalPos == StringRef::npos) {
      continue;
    }
    auto *name = allocator.Allocate<char>(equalPos + 1);
    std::copy(ptr, ptr + equalPos, name);
    name[equalPos] = '\0';
    store.push_back(fmt::arg<char>(name, env.substr(equalPos + 1).data()));
  }
  return store;
}
} // namespace
StringRef interpolate_config_string(llvm::StringRef str,
                                    BumpPtrAllocator &allocator) {
  static EnvArgStore envArgStore = create_interpolate_arg_store(allocator);
  auto res =
      fmt::vformat(std::string_view(str.data(), str.size()), envArgStore);
  if (res == str) {
    return str;
  }
  return StringRef(res).copy(allocator);
}

struct AggregatedExporter {
  FFIExporterContext context;
  AggregatedExporter(FFIExporterContext context) : context(context) {}
  std::vector<std::unique_ptr<FFIExporter>> exporters;
  ExportCAPIConsumer &getConsumer() {
    return *(ExportCAPIConsumer *)context.impl;
  }
  void add_language(StringRef langName, StringRef outputPath,
                    const YAML::Node &langConf);
  template <typename F> bool exportForwardImpl(F &&func) {
    bool ok = true;
    for (auto &exporter : make_pointee_range(exporters)) {
      if (!func(exporter)) {
        ok = false;
      }
    }
    return ok;
  }
  bool exportTypeDefNameDecl(TypedefNameDecl *typedefDecl) {
    return exportForwardImpl([typedefDecl](FFIExporter &exporter) {
      return exporter.exportTypeDefNameDecl(typedefDecl);
    });
  }
  bool exportEnumDecl(EnumDecl *enumDecl) {
    return exportForwardImpl([enumDecl](FFIExporter &exporter) {
      return exporter.exportEnumDecl(enumDecl);
    });
  }
  bool exportRecordDecl(RecordDecl *recordDecl) {
    return exportForwardImpl([recordDecl](FFIExporter &exporter) {
      return exporter.exportRecordDecl(recordDecl);
    });
  }
  bool exportFunctionDecl(FunctionDecl *funcDecl) {
    return exportForwardImpl([funcDecl](FFIExporter &exporter) {
      return exporter.exportFunctionDecl(funcDecl);
    });
  }
  void finishTranslationUnit(ASTContext &Ctx) {
    exportForwardImpl([&](FFIExporter &exporter) {
      exporter.finishTranslationUnit(Ctx);
      return true;
    });
  }
  void handleMainFileInclude(StringRef fileName) {
    exportForwardImpl([fileName](FFIExporter &exporter) {
      exporter.handleMainFileInclude(fileName);
      return true;
    });
  }
};

class ExportCAPIConsumer : public ASTConsumer {
  friend class FFIExporter;
  BumpPtrAllocator &allocator;
  CompilerInstance &compInst;
  YAML::Node config;
  AggregatedExporter exporter;
  bool hasError = false;
  ASTContext *astContext = nullptr;

public:
  ExportCAPIConsumer(CompilerInstance &compInst, YAML::Node &&config,
                     const StringMap<std::string> &langOutputs,
                     BumpPtrAllocator &allocator)
      : compInst(compInst), config(std::move(config)), allocator(allocator),
        exporter(FFIExporterContext{this}) {
    const auto &langs = config["languages"];
    for (const auto &[lang, path] : langOutputs) {
      exporter.add_language(lang, path, langs[lang.str()]);
    }
  }
  void diagError(Twine message) {
    hasError = true;
    SmallVector<char> buffer;
    auto &Diag = compInst.getDiagnostics();
    unsigned DiagID = Diag.getDiagnosticIDs()->getCustomDiagID(
        DiagnosticIDs::Error,
        (StringRef("[CAPI Exporter] ") + message).toStringRef(buffer));
    Diag.Report(DiagID);
  }
  //  std::unique_ptr<llvm::raw_ostream> cs_, cpp_;
  //  CompilerInstance &compInst;
  //  ASTContext *context = nullptr;
  //  llvm::DenseSet<const Type *> visitedTypes;
  //  unsigned indentLevel = 0;
  //  LangOptions refLangOpt;
  //  PrintingPolicy refPrintPolicy{refLangOpt};
  //  struct IndentGuard {
  //    ExportCAPIConsumer *parent;
  //    unsigned level;
  //    IndentGuard(ExportCAPIAction *parent, unsigned level)
  //        : parent(parent), level(level) {
  //      parent->indentLevel += level;
  //    }
  //    ~IndentGuard() { parent->indentLevel -= level; }
  //  };
  //  llvm::raw_ostream &cs() { return *cs_; }
  //  llvm::raw_ostream &cpp() { return *cpp_; }
  //  void indent(llvm::raw_ostream &os) { os.indent(indentLevel); }
  //  bool hasName(NamedDecl *decl) { return decl->getDeclName().isIdentifier();
  //  } void emitTypeReference(QualType qualType) {
  //    emitTypeReference(qualType.getTypePtrOrNull());
  //  }
  //  void emitTypeReference(const Type *type) {
  //    if (auto *parenType = dyn_cast<ParenType>(type)) {
  //      return emitTypeReference(parenType->desugar());
  //    }
  //    if (auto *elabType = dyn_cast<ElaboratedType>(type)) {
  //      return emitTypeReference(elabType->desugar());
  //    }
  //    if (auto *recordType = dyn_cast<RecordType>(type)) {
  //      auto str = QualType(type, {}).getAsString();
  //      str = std::regex_replace(str, std::regex("^struct "), "struct-");
  //      cs() << str;
  //      return;
  //    }
  //    if (auto *enumType = dyn_cast<EnumType>(type)) {
  //      auto str = QualType(type, {}).getAsString();
  //      str = std::regex_replace(str, std::regex("^enum "), "enum-");
  //      cs() << str;
  //      return;
  //    }
  //    if (auto *typedefType = dyn_cast<TypedefType>(type)) {
  //      auto innerType = typedefType->desugar();
  //      if (isa<BuiltinType, TypedefType, ElaboratedType>(innerType)) {
  //        emitTypeReference(innerType);
  //        return;
  //      }
  //      auto *typedefDecl = typedefType->getDecl();
  //      cs() << typedefDecl->getName();
  //      return;
  //    }
  //    if (auto *builtinType = dyn_cast<BuiltinType>(type)) {
  //      if (type->isVoidType()) {
  //        cs() << "void";
  //        return;
  //      }
  //      if (type->isIntegralType(*context)) {
  //        if (type->isSignedIntegerType()) {
  //          cs() << "integer-";
  //          cs() << context->getTypeSizeInChars(type).getQuantity() * 8;
  //          return;
  //        }
  //        cs() << "unsigned-";
  //        cs() << context->getTypeSizeInChars(type).getQuantity() * 8;
  //        return;
  //      }
  //      if (builtinType->isFloatingPoint()) {
  //        auto str = QualType(type, {}).getAsString();
  //        cs() << str;
  //        return;
  //      }
  //      {
  //        auto &Diag = compInst.getDiagnostics();
  //        unsigned DiagID = Diag.getCustomDiagID(
  //            DiagnosticsEngine::Error,
  //            "unhandled BuiltinType in emitTypeReference: %0");
  //        std::string str;
  //        llvm::raw_string_ostream sos(str);
  //        type->dump(sos, *context);
  //        Diag.Report(DiagID) << str;
  //      }
  //      return;
  //    }
  //    emitTypeBody(type);
  //  }
  //  void emitTypeBody(QualType qualType) {
  //    emitTypeBody(qualType.getTypePtrOrNull());
  //  }
  //  void emitTypeBody(const Type *type) {
  //    if (auto *parenType = dyn_cast<ParenType>(type)) {
  //      return emitTypeBody(parenType->desugar());
  //    }
  //    if (auto *elabType = dyn_cast<ElaboratedType>(type)) {
  //      return emitTypeBody(elabType->desugar());
  //    }
  //    if (auto *recordType = dyn_cast<RecordType>(type)) {
  //      emitRecordTypeBody(recordType);
  //    }
  //    if (auto *funcType = dyn_cast<FunctionProtoType>(type)) {
  //      cs() << "(function \n";
  //      {
  //        IndentGuard g(this, 2);
  //        indent(cs());
  //        cs() << "(";
  //        bool isFirst = true;
  //        for (auto paramType : funcType->getParamTypes()) {
  //          if (!isFirst) {
  //            cs() << ' ';
  //          }
  //          isFirst = false;
  //          emitTypeReference(paramType);
  //        }
  //        cs() << ")\n";
  //        indent(cs());
  //        emitTypeReference(funcType->getReturnType());
  //      }
  //      cs() << ")";
  //      return;
  //    }
  //    if (auto *ptrType = dyn_cast<PointerType>(type)) {
  //      cs() << "(* ";
  //      emitTypeReference(type->getPointeeType());
  //      cs() << ")";
  //      return;
  //    }
  //    auto &Diag = compInst.getDiagnostics();
  //    unsigned DiagID = Diag.getCustomDiagID(
  //        DiagnosticsEngine::Error, "unhandled type in emitTypeBody: %0");
  //    std::string str;
  //    llvm::raw_string_ostream sos(str);
  //    type->dump(sos, *context);
  //    Diag.Report(DiagID) << str;
  //  }
  //  void emitRecordTypeBody(const RecordType *recordType) {
  //    auto *recordDecl = recordType->getDecl();
  //    indent(cs());
  //    cs() << "(struct\n";
  //    {
  //      IndentGuard g(this, 2);
  //      for (auto *fieldDecl : recordDecl->fields()) {
  //        indent(cs());
  //        cs() << "(";
  //        fieldDecl->printName(cs(), refPrintPolicy);
  //        cs() << " ";
  //        emitTypeReference(fieldDecl->getType());
  //        cs() << ")\n";
  //      }
  //    }
  //    indent(cs());
  //    cs() << ")\n";
  //  }
  //  void ensureTypeEmitted(QualType qualType) {
  //    ensureTypeEmitted(qualType.getTypePtrOrNull());
  //  }
  //  void ensureTypeEmitted(const Type *type) {
  //    if (!type) {
  //      return;
  //    }
  //    if (!visitedTypes.insert(type).second) {
  //      return;
  //    }
  //    if (auto *recordType = dyn_cast<RecordType>(type)) {
  //      if (auto *recordDecl = recordType->getDecl()) {
  //        emitRecordDecl(recordDecl);
  //      }
  //      return;
  //    }
  //    if (auto *typedefType = dyn_cast<TypedefType>(type)) {
  //      if (auto *typedefDecl = typedefType->getDecl()) {
  //        emitTypedefDecl(typedefDecl);
  //      }
  //      return;
  //    }
  //  }
  //  void emitRecordDecl(RecordDecl *recordDecl) {
  //    if (!hasName(recordDecl) || !isInMainFile(recordDecl)) {
  //      return;
  //    }
  //    if (!recordDecl->isThisDeclarationADefinition()) {
  //      return;
  //    }
  //    auto *type = context->getTypeDeclType(recordDecl).getTypePtrOrNull();
  //    auto *recordType = cast<RecordType>(type);
  //    unsigned numFields = 0;
  //    for (auto *fieldDecl : recordDecl->fields()) {
  //      ensureTypeEmitted(fieldDecl->getType());
  //      ++numFields;
  //    }
  //    indent(cs());
  //    cs() << "(define-ftype ";
  //    emitTypeReference(recordType);
  //    cs() << "\n";
  //    {
  //      IndentGuard g(this, 2);
  //      emitRecordTypeBody(recordType);
  //    }
  //    cs() << ")\n";
  //  }
  //  void emitTypedefDecl(TypedefNameDecl *typedefDecl) {
  //    if (!hasName(typedefDecl) || !isInMainFile(typedefDecl)) {
  //      return;
  //    }
  //    auto *definedType = typedefDecl->getUnderlyingType().getTypePtrOrNull();
  //    if (!definedType) {
  //      return;
  //    }
  //    indent(cs());
  //    cs() << "(define-ftype " << typedefDecl->getName() << " ";
  //    emitTypeReference(definedType);
  //    cs() << ")\n";
  //  }
  //  const Type *desugarAll(const Type *type) {
  //    const Type *newType = type;
  //    do {
  //      type = newType;
  //      if (auto *parenType = dyn_cast<ParenType>(newType)) {
  //        newType = parenType->desugar().getTypePtrOrNull();
  //      }
  //      if (auto *elabType = dyn_cast<ElaboratedType>(newType)) {
  //        newType = elabType->desugar().getTypePtrOrNull();
  //      }
  //      if (auto *typedefType = dyn_cast<TypedefType>(newType)) {
  //        newType = typedefType->desugar().getTypePtrOrNull();
  //      }
  //    } while (newType != type);
  //    return newType;
  //  }
  //  const Type *removeQual(QualType qualType) {
  //    return qualType.getTypePtrOrNull();
  //  }
  //  const Type *removeQual(const Type *type) { return type; }
  //  const Type *getPassArgumentType(QualType qualType) {
  //    return getPassArgumentType(qualType.getTypePtrOrNull());
  //  }
  //  const Type *getPassArgumentType(const Type *type) {
  //    auto *innerType = desugarAll(type);
  //    if (isa<RecordType>(innerType)) {
  //      return context->getPointerType(QualType(type, {})).getTypePtrOrNull();
  //    }
  //    return type;
  //  }
  //  void emitFunctionDecl(FunctionDecl *funcDecl) {
  //    if (!isInMainFile(funcDecl)) {
  //      return;
  //    }
  //    auto *funcType = dyn_cast_or_null<FunctionProtoType>(
  //        funcDecl->getType().getTypePtrOrNull());
  //    if (!funcType) {
  //      return;
  //    }
  //    auto *retType = removeQual(funcType->getReturnType());
  //    ensureTypeEmitted(retType);
  //    auto *retStubType = getPassArgumentType(retType);
  //    bool needStub = retType != retStubType;
  //    llvm::SmallVector<const Type *> paramStubTypes;
  //    auto paramTypes = funcType->getParamTypes();
  //    for (auto paramTy_ : paramTypes) {
  //      auto *paramType = removeQual(paramTy_);
  //      ensureTypeEmitted(paramType);
  //      auto *paramStubType = getPassArgumentType(paramType);
  //      needStub = needStub || paramType != paramStubType;
  //      paramStubTypes.push_back(paramStubType);
  //    }
  //    indent(cs());
  //    auto realFuncName = funcDecl->getName();
  //    llvm::Twine funcName = needStub ? realFuncName + "_stub" : realFuncName;
  //    if (needStub) {
  //      bool hasRetArg = false;
  //      bool isFirstArg = true;
  //      if (retStubType != retType) {
  //        cpp() << "void " << funcName << "(";
  //        QualType::print(retStubType, {}, cpp(), refPrintPolicy, "ret");
  //        isFirstArg = false;
  //        hasRetArg = true;
  //      } else {
  //        QualType::print(retStubType, {}, cpp(), refPrintPolicy, "");
  //        cpp() << " " << funcName << "(";
  //      }
  //      for (size_t i = 0, e = paramStubTypes.size(); i < e; ++i) {
  //        if (!isFirstArg) {
  //          cpp() << ", ";
  //        }
  //        isFirstArg = false;
  //        QualType::print(paramStubTypes[i], {}, cpp(), refPrintPolicy,
  //                        "arg" + std::to_string(i));
  //      }
  //      cpp() << ") {\n";
  //      cpp() << "  ";
  //      if (hasRetArg) {
  //        cpp() << "*ret = ";
  //      } else {
  //        cpp() << "return ";
  //      }
  //      cpp() << realFuncName << "(";
  //      for (size_t i = 0, e = paramStubTypes.size(); i < e; ++i) {
  //        if (paramStubTypes[i] == removeQual(paramTypes[i])) {
  //          cpp() << "arg" << i;
  //        } else {
  //          cpp() << "*arg" << i;
  //        }
  //        if (i + 1 != e) {
  //          cpp() << ", ";
  //        }
  //      }
  //      cpp() << ");\n";
  //      cpp() << "}\n";
  //    }
  //    cs() << "(define " << funcName << '\n';
  //    {
  //      IndentGuard guard(this, 2);
  //      indent(cs());
  //      cs() << "(foreign-procedure " << "\"" << funcName << "\"\n";
  //      {
  //        IndentGuard guard(this, 2);
  //        indent(cs());
  //        cs() << "(";
  //        bool isFirst = true;
  //        for (auto paramTy : funcType->getParamTypes()) {
  //          if (!isFirst) {
  //            cs() << ' ';
  //          }
  //          isFirst = false;
  //          emitTypeReference(paramTy);
  //        }
  //        cs() << ")\n";
  //        indent(cs());
  //        emitTypeReference(funcType->getReturnType());
  //        cs() << ")";
  //      }
  //    }
  //    cs() << ")\n";
  //  }
  //
  // public:
  //  ExportCAPIConsumer(std::unique_ptr<llvm::raw_ostream> cs_,
  //                    std::unique_ptr<llvm::raw_ostream> cpp_,
  //                    CompilerInstance &compInst)
  //      : cs_(std::move(cs_)), cpp_(std::move(cpp_)), compInst(compInst) {
  //    auto &srcMgr = compInst.getSourceManager();
  //    auto mainFileId = srcMgr.getMainFileID();
  //    auto *mainFileEntry = srcMgr.getFileEntryForID(mainFileId);
  //    if (!mainFileEntry) {
  //      auto &Diag = compInst.getDiagnostics();
  //      unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
  //                                             "can not get main file path");
  //      Diag.Report(DiagID);
  //      return;
  //    }
  //    cpp() << "#include \"" << mainFileEntry->tryGetRealPathName() <<
  //    "\"\n\n";
  //    // FIXME:
  //    cpp() << "#include \"" << "mlir-c/AffineMap.h" << "\"\n\n";
  //  }
  bool addFunctionDecl(FunctionDecl *funcDecl) {
    exporter.exportFunctionDecl(funcDecl);
    return true;
  }
  bool addRecordDecl(RecordDecl *recordDecl) {
    if (!recordDecl->isThisDeclarationADefinition()) {
      return true;
    }
    exporter.exportRecordDecl(recordDecl);
    return true;
  }
  bool addTypeDefDecl(TypedefNameDecl *typedefDecl) {
    return exporter.exportTypeDefNameDecl(typedefDecl);
  }
  bool addEnumDecl(EnumDecl *enumDecl) {
    return exporter.exportEnumDecl(enumDecl);
  }
  void emitDiag(Twine message) {
    SmallVector<char> buffer;
    auto &Diag = compInst.getDiagnostics();
    unsigned DiagID = Diag.getDiagnosticIDs()->getCustomDiagID(
        DiagnosticIDs::Error,
        (StringRef("[CAPI Exporter] ") + message).toStringRef(buffer));
    Diag.Report(DiagID);
  }
  bool filterSourceLocation(SourceLocation loc) {
    auto &srcMgr = compInst.getSourceManager();
    // FIXME: what if no spelling loc: FOO(BAR), should we get the root of
    // expansion loc?
    loc = srcMgr.getExpansionLoc(loc);
    auto fileId = srcMgr.getFileID(loc);
    if (fileId == srcMgr.getMainFileID()) {
      return true;
    }
    auto *fileRef = srcMgr.getFileEntryForID(fileId);
    if (!fileRef) {
      return false;
    }
    auto filePath = fileRef->tryGetRealPathName();
    if (filePath.ends_with(".def") || filePath.ends_with(".inc")) {
      return true;
    }
    return false;
  }
  bool handleLinkageSpecDecl(LinkageSpecDecl *linkageSpec) {
    for (auto *decl : linkageSpec->decls()) {
      if (!handleDecl(decl)) {
        return false;
      }
    }
    return true;
  }
  bool handleDecl(Decl *decl) {
    if (auto *linkageSpec = dyn_cast<LinkageSpecDecl>(decl)) {
      return handleLinkageSpecDecl(linkageSpec);
    }
    if (!filterSourceLocation(decl->getLocation())) {
      return true;
    }
    if (auto *funcDecl = dyn_cast<FunctionDecl>(decl)) {
      return addFunctionDecl(funcDecl);
    }
    if (auto *recordDecl = dyn_cast<RecordDecl>(decl)) {
      return addRecordDecl(recordDecl);
    }
    if (auto *typedefDecl = dyn_cast<TypedefNameDecl>(decl)) {
      return addTypeDefDecl(typedefDecl);
    }
    if (auto *enumDecl = dyn_cast<EnumDecl>(decl)) {
      return addEnumDecl(enumDecl);
    }
    decl->dump();
    emitDiag("unknown decl");
    return true;
  }
  virtual void Initialize(ASTContext &Context) override {
    this->astContext = &Context;
  }
  bool HandleTopLevelDecl(DeclGroupRef D) override {
    for (auto *decl : D) {
      if (!handleDecl(decl)) {
        return false;
      }
    }
    return true;
  }
  void HandleTranslationUnit(ASTContext &Ctx) override {
    exporter.finishTranslationUnit(Ctx);
  }

  void handleMainFileInclude(StringRef fileName) {
    exporter.handleMainFileInclude(fileName);
  }
};

void AggregatedExporter::add_language(StringRef langName, StringRef outputPath,
                                      const YAML::Node &langConf) {
  if (langName == "rust") {
    exporters.emplace_back(createRustExporter(outputPath, langConf, context));
  } else if (langName == "chez-scheme") {
    exporters.emplace_back(
        createChezSchemeExporter(outputPath, langConf, context));
  } else {
    getConsumer().diagError(Twine("unknown export language ") + langName);
  }
}

struct WatchIncludeCallbacks : PPCallbacks {
  CompilerInstance &compInst;
  SourceManager *srcMgr;
  ExportCAPIConsumer *consumer = nullptr;
  WatchIncludeCallbacks(CompilerInstance &compInst) : compInst(compInst) {
    srcMgr = &compInst.getSourceManager();
  }

  //  // NB: This doens't work when A includes B and C, and B includes C. We
  //  won't
  //  // get A's inclusion of C.
  //  void LexedFileChanged(FileID fID, LexedFileChangeReason reason,
  //                        SrcMgr::CharacteristicKind fileType, FileID prevFID,
  //                        SourceLocation loc) override {
  //    if (prevFID != srcMgr->getMainFileID()) {
  //      return;
  //    }
  //    auto fName = srcMgr->getNonBuiltinFilenameForID(fID);
  //    if (fName) {
  //      llvm::dbgs() << "lex change: " << fName << '\n';
  //    }
  //  }
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          OptionalFileEntryRef File, StringRef SearchPath,
                          StringRef RelativePath,
                          const clang::Module *SuggestedModule,
                          bool ModuleImported,
                          SrcMgr::CharacteristicKind FileType) override {
    // auto prevFID = srcMgr->getFileID(HashLoc);
    // if (prevFID != srcMgr->getMainFileID()) {
    //   return;
    // }
    //  TODO: rename
    consumer->handleMainFileInclude(FileName);
  }
  void setConsumer(ExportCAPIConsumer *consumer) { this->consumer = consumer; }
};

class ExportCAPIAction : public PluginASTAction {
  BumpPtrAllocator allocator;
  YAML::Node config;
  StringMap<std::string> langOutputs;
  WatchIncludeCallbacks *ppCb;

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    auto res = std::make_unique<ExportCAPIConsumer>(CI, std::move(config),
                                                    langOutputs, allocator);
    ppCb->setConsumer(res.get());
    return res;
  }
  bool BeginSourceFileAction(CompilerInstance &CI) override {
    auto cb = std::make_unique<WatchIncludeCallbacks>(CI);
    ppCb = cb.get();
    CI.getPreprocessor().addPPCallbacks(std::move(cb));
    return true;
  }
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    auto emitDiag = [&](Twine message) {
      SmallVector<char> buffer;
      auto &Diag = CI.getDiagnostics();
      unsigned DiagID = Diag.getDiagnosticIDs()->getCustomDiagID(
          DiagnosticIDs::Error,
          (StringRef("[CAPI Exporter] ") + message).toStringRef(buffer));
      Diag.Report(DiagID);
      return false;
    };
    auto emitExpectArgError = [&](size_t pos) {
      return emitDiag(Twine("option") + args[pos] + " expects an argument");
    };
    static const char *known_langs[] = {"rust", "chez_scheme"};
    const char *confPathStr = nullptr;
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      const auto &arg = args[i];
      const char *findLang = nullptr;
      for (const auto *lang : known_langs) {
        if (arg == lang) {
          findLang = lang;
          break;
        }
      }
      if (findLang) {
        if (i + 1 == e) {
          return emitExpectArgError(i);
        }
        langOutputs[findLang] = args[++i];
        continue;
      }
      if (confPathStr) {
        return emitDiag(Twine("config file specified more than once, first: ") +
                        confPathStr + StringRef(" then: ") + arg);
      }
      confPathStr = arg.data();
    }
    if (!confPathStr) {
      return emitDiag("must provide config file");
    }
    try {
      StringRef confPath;
      try {
        confPath = interpolate_config_string(confPathStr, allocator);
      } catch (const std::exception &e) {
        return emitDiag(Twine("failed to get confg path: ") + e.what());
      }
      config = YAML::LoadFile(confPath.str());
    } catch (const YAML::BadFile &badFile) {
      return emitDiag(Twine("can not open config file: ") + badFile.what());
    } catch (const YAML::ParserException &parserErr) {
      return emitDiag(Twine("can not parse config file: ") + parserErr.what());
    } catch (const std::exception &e) {
      return emitDiag(Twine("unknown error: ") + e.what());
    }
    return true;
  }
};

ASTContext &FFIExporter::getASTContext() { return *getConsumer().astContext; }

DiagnosticsEngine &FFIExporter::getDiagnostics() {
  return getConsumer().compInst.getDiagnostics();
}

ExportCAPIConsumer &FFIExporter::getConsumer() {
  return *(ExportCAPIConsumer *)context.impl;
}

void FFIExporter::diagError(Twine message_) {
  hasError = true;
  Twine message =
      lang.empty() ? message_ : Twine("[") + lang + " exporter] " + message_;
  return getConsumer().diagError(message);
}

FFIExporter::~FFIExporter() {}

} // namespace capi_ffi_gen

static FrontendPluginRegistry::Add<capi_ffi_gen::ExportCAPIAction>
    X("capi_ffi_gen", "export CAPI through other languages' FFI");
