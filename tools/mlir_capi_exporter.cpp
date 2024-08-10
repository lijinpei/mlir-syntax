#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/raw_ostream.h"

#include <regex>

using namespace clang;

namespace {

namespace MLIR_CAPI_ChezSchemeFFI {
class ExportFFIConsumer : public ASTConsumer {
  std::unique_ptr<llvm::raw_ostream> cs_, cpp_;
  CompilerInstance &compInst;
  ASTContext *context = nullptr;
  llvm::DenseSet<const Type *> visitedTypes;
  unsigned indentLevel = 0;
  LangOptions refLangOpt;
  PrintingPolicy refPrintPolicy{refLangOpt};
  struct IndentGuard {
    ExportFFIConsumer *parent;
    unsigned level;
    IndentGuard(ExportFFIConsumer *parent, unsigned level)
        : parent(parent), level(level) {
      parent->indentLevel += level;
    }
    ~IndentGuard() { parent->indentLevel -= level; }
  };
  llvm::raw_ostream &cs() { return *cs_; }
  llvm::raw_ostream &cpp() { return *cpp_; }
  void indent(llvm::raw_ostream &os) { os.indent(indentLevel); }
  bool isInMainFile(Decl *decl) {
    auto &srcMgr = compInst.getSourceManager();
    return srcMgr.getFileID(decl->getLocation()) == srcMgr.getMainFileID();
  }
  bool hasName(NamedDecl *decl) { return decl->getDeclName().isIdentifier(); }
  void emitTypeReference(QualType qualType) {
    emitTypeReference(qualType.getTypePtrOrNull());
  }
  void emitTypeReference(const Type *type) {
    if (auto *parenType = dyn_cast<ParenType>(type)) {
      return emitTypeReference(parenType->desugar());
    }
    if (auto *elabType = dyn_cast<ElaboratedType>(type)) {
      return emitTypeReference(elabType->desugar());
    }
    if (auto *recordType = dyn_cast<RecordType>(type)) {
      auto str = QualType(type, {}).getAsString();
      str = std::regex_replace(str, std::regex("^struct "), "struct-");
      cs() << str;
      return;
    }
    if (auto *enumType = dyn_cast<EnumType>(type)) {
      auto str = QualType(type, {}).getAsString();
      str = std::regex_replace(str, std::regex("^enum "), "enum-");
      cs() << str;
      return;
    }
    if (auto *typedefType = dyn_cast<TypedefType>(type)) {
      auto innerType = typedefType->desugar();
      if (isa<BuiltinType, TypedefType, ElaboratedType>(innerType)) {
        emitTypeReference(innerType);
        return;
      }
      auto *typedefDecl = typedefType->getDecl();
      cs() << typedefDecl->getName();
      return;
    }
    if (auto *builtinType = dyn_cast<BuiltinType>(type)) {
      if (type->isVoidType()) {
        cs() << "void";
        return;
      }
      if (type->isIntegralType(*context)) {
        if (type->isSignedIntegerType()) {
          cs() << "integer-";
          cs() << context->getTypeSizeInChars(type).getQuantity() * 8;
          return;
        }
        cs() << "unsigned-";
        cs() << context->getTypeSizeInChars(type).getQuantity() * 8;
        return;
      }
      if (builtinType->isFloatingPoint()) {
        auto str = QualType(type, {}).getAsString();
        cs() << str;
        return;
      }
      {
        auto &Diag = compInst.getDiagnostics();
        unsigned DiagID = Diag.getCustomDiagID(
            DiagnosticsEngine::Error,
            "unhandled BuiltinType in emitTypeReference: %0");
        std::string str;
        llvm::raw_string_ostream sos(str);
        type->dump(sos, *context);
        Diag.Report(DiagID) << str;
      }
      return;
    }
    emitTypeBody(type);
  }
  void emitTypeBody(QualType qualType) {
    emitTypeBody(qualType.getTypePtrOrNull());
  }
  void emitTypeBody(const Type *type) {
    if (auto *parenType = dyn_cast<ParenType>(type)) {
      return emitTypeBody(parenType->desugar());
    }
    if (auto *elabType = dyn_cast<ElaboratedType>(type)) {
      return emitTypeBody(elabType->desugar());
    }
    if (auto *recordType = dyn_cast<RecordType>(type)) {
      emitRecordTypeBody(recordType);
    }
    if (auto *funcType = dyn_cast<FunctionProtoType>(type)) {
      cs() << "(function \n";
      {
        IndentGuard g(this, 2);
        indent(cs());
        cs() << "(";
        bool isFirst = true;
        for (auto paramType : funcType->getParamTypes()) {
          if (!isFirst) {
            cs() << ' ';
          }
          isFirst = false;
          emitTypeReference(paramType);
        }
        cs() << ")\n";
        indent(cs());
        emitTypeReference(funcType->getReturnType());
      }
      cs() << ")";
      return;
    }
    if (auto *ptrType = dyn_cast<PointerType>(type)) {
      cs() << "(* ";
      emitTypeReference(type->getPointeeType());
      cs() << ")";
      return;
    }
    auto &Diag = compInst.getDiagnostics();
    unsigned DiagID = Diag.getCustomDiagID(
        DiagnosticsEngine::Error, "unhandled type in emitTypeBody: %0");
    std::string str;
    llvm::raw_string_ostream sos(str);
    type->dump(sos, *context);
    Diag.Report(DiagID) << str;
  }
  void emitRecordTypeBody(const RecordType *recordType) {
    auto *recordDecl = recordType->getDecl();
    indent(cs());
    cs() << "(struct\n";
    {
      IndentGuard g(this, 2);
      for (auto *fieldDecl : recordDecl->fields()) {
        indent(cs());
        cs() << "(";
        fieldDecl->printName(cs(), refPrintPolicy);
        cs() << " ";
        emitTypeReference(fieldDecl->getType());
        cs() << ")\n";
      }
    }
    indent(cs());
    cs() << ")\n";
  }
  void ensureTypeEmitted(QualType qualType) {
    ensureTypeEmitted(qualType.getTypePtrOrNull());
  }
  void ensureTypeEmitted(const Type *type) {
    if (!type) {
      return;
    }
    if (!visitedTypes.insert(type).second) {
      return;
    }
    if (auto *recordType = dyn_cast<RecordType>(type)) {
      if (auto *recordDecl = recordType->getDecl()) {
        emitRecordDecl(recordDecl);
      }
      return;
    }
    if (auto *typedefType = dyn_cast<TypedefType>(type)) {
      if (auto *typedefDecl = typedefType->getDecl()) {
        emitTypedefDecl(typedefDecl);
      }
      return;
    }
  }
  void emitRecordDecl(RecordDecl *recordDecl) {
    if (!hasName(recordDecl) || !isInMainFile(recordDecl)) {
      return;
    }
    if (!recordDecl->isThisDeclarationADefinition()) {
      return;
    }
    auto *type = context->getTypeDeclType(recordDecl).getTypePtrOrNull();
    auto *recordType = cast<RecordType>(type);
    unsigned numFields = 0;
    for (auto *fieldDecl : recordDecl->fields()) {
      ensureTypeEmitted(fieldDecl->getType());
      ++numFields;
    }
    indent(cs());
    cs() << "(define-ftype ";
    emitTypeReference(recordType);
    cs() << "\n";
    {
      IndentGuard g(this, 2);
      emitRecordTypeBody(recordType);
    }
    cs() << ")\n";
  }
  void emitTypedefDecl(TypedefNameDecl *typedefDecl) {
    if (!hasName(typedefDecl) || !isInMainFile(typedefDecl)) {
      return;
    }
    auto *definedType = typedefDecl->getUnderlyingType().getTypePtrOrNull();
    if (!definedType) {
      return;
    }
    indent(cs());
    cs() << "(define-ftype " << typedefDecl->getName() << " ";
    emitTypeReference(definedType);
    cs() << ")\n";
  }
  const Type *desugarAll(const Type *type) {
    const Type *newType = type;
    do {
      type = newType;
      if (auto *parenType = dyn_cast<ParenType>(newType)) {
        newType = parenType->desugar().getTypePtrOrNull();
      }
      if (auto *elabType = dyn_cast<ElaboratedType>(newType)) {
        newType = elabType->desugar().getTypePtrOrNull();
      }
      if (auto *typedefType = dyn_cast<TypedefType>(newType)) {
        newType = typedefType->desugar().getTypePtrOrNull();
      }
    } while (newType != type);
    return newType;
  }
  const Type *removeQual(QualType qualType) {
    return qualType.getTypePtrOrNull();
  }
  const Type *removeQual(const Type *type) { return type; }
  const Type *getPassArgumentType(QualType qualType) {
    return getPassArgumentType(qualType.getTypePtrOrNull());
  }
  const Type *getPassArgumentType(const Type *type) {
    auto *innerType = desugarAll(type);
    if (isa<RecordType>(innerType)) {
      return context->getPointerType(QualType(type, {})).getTypePtrOrNull();
    }
    return type;
  }
  void emitFunctionDecl(FunctionDecl *funcDecl) {
    if (!isInMainFile(funcDecl)) {
      return;
    }
    auto *funcType = dyn_cast_or_null<FunctionProtoType>(
        funcDecl->getType().getTypePtrOrNull());
    if (!funcType) {
      return;
    }
    auto *retType = removeQual(funcType->getReturnType());
    ensureTypeEmitted(retType);
    auto *retStubType = getPassArgumentType(retType);
    bool needStub = retType != retStubType;
    llvm::SmallVector<const Type *> paramStubTypes;
    auto paramTypes = funcType->getParamTypes();
    for (auto paramTy_ : paramTypes) {
      auto *paramType = removeQual(paramTy_);
      ensureTypeEmitted(paramType);
      auto *paramStubType = getPassArgumentType(paramType);
      needStub = needStub || paramType != paramStubType;
      paramStubTypes.push_back(paramStubType);
    }
    indent(cs());
    auto realFuncName = funcDecl->getName();
    llvm::Twine funcName = needStub ? realFuncName + "_stub" : realFuncName;
    if (needStub) {
      bool hasRetArg = false;
      bool isFirstArg = true;
      if (retStubType != retType) {
        cpp() << "void " << funcName << "(";
        QualType::print(retStubType, {}, cpp(), refPrintPolicy, "ret");
        isFirstArg = false;
        hasRetArg = true;
      } else {
        QualType::print(retStubType, {}, cpp(), refPrintPolicy, "");
        cpp() << " " << funcName << "(";
      }
      for (size_t i = 0, e = paramStubTypes.size(); i < e; ++i) {
        if (!isFirstArg) {
          cpp() << ", ";
        }
        isFirstArg = false;
        QualType::print(paramStubTypes[i], {}, cpp(), refPrintPolicy,
                        "arg" + std::to_string(i));
      }
      cpp() << ") {\n";
      cpp() << "  ";
      if (hasRetArg) {
        cpp() << "*ret = ";
      } else {
        cpp() << "return ";
      }
      cpp() << realFuncName << "(";
      for (size_t i = 0, e = paramStubTypes.size(); i < e; ++i) {
        if (paramStubTypes[i] == removeQual(paramTypes[i])) {
          cpp() << "arg" << i;
        } else {
          cpp() << "*arg" << i;
        }
        if (i + 1 != e) {
          cpp() << ", ";
        }
      }
      cpp() << ");\n";
      cpp() << "}\n";
    }
    cs() << "(define " << funcName << '\n';
    {
      IndentGuard guard(this, 2);
      indent(cs());
      cs() << "(foreign-procedure " << "\"" << funcName << "\"\n";
      {
        IndentGuard guard(this, 2);
        indent(cs());
        cs() << "(";
        bool isFirst = true;
        for (auto paramTy : funcType->getParamTypes()) {
          if (!isFirst) {
            cs() << ' ';
          }
          isFirst = false;
          emitTypeReference(paramTy);
        }
        cs() << ")\n";
        indent(cs());
        emitTypeReference(funcType->getReturnType());
        cs() << ")";
      }
    }
    cs() << ")\n";
  }
  void handleLinkageSpecDecl(LinkageSpecDecl *linkageSpec) {
    for (auto *decl : linkageSpec->decls()) {
      handleDecl(decl);
    }
  }
  void handleDecl(Decl *decl) {
    if (auto *linkageSpec = dyn_cast<LinkageSpecDecl>(decl)) {
      return handleLinkageSpecDecl(linkageSpec);
    }
    if (auto *funcDecl = dyn_cast<FunctionDecl>(decl)) {
      return emitFunctionDecl(funcDecl);
    }
    if (auto *recordDecl = dyn_cast<RecordDecl>(decl)) {
      return emitRecordDecl(recordDecl);
    }
    if (auto *typedefDecl = dyn_cast<TypedefNameDecl>(decl)) {
      return emitTypedefDecl(typedefDecl);
    }
  }
  void handleDeclGroup(DeclGroupRef DG) {
    for (auto *decl : DG) {
      handleDecl(decl);
    }
  }

public:
  ExportFFIConsumer(std::unique_ptr<llvm::raw_ostream> cs_,
                    std::unique_ptr<llvm::raw_ostream> cpp_,
                    CompilerInstance &compInst)
      : cs_(std::move(cs_)), cpp_(std::move(cpp_)), compInst(compInst) {
    auto &srcMgr = compInst.getSourceManager();
    auto mainFileId = srcMgr.getMainFileID();
    auto *mainFileEntry = srcMgr.getFileEntryForID(mainFileId);
    if (!mainFileEntry) {
      auto &Diag = compInst.getDiagnostics();
      unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                             "can not get main file path");
      Diag.Report(DiagID);
      return;
    }
    cpp() << "#include \"" << mainFileEntry->tryGetRealPathName() << "\"\n\n";
    // FIXME:
    cpp() << "#include \"" << "mlir-c/AffineMap.h" << "\"\n\n";
  }
  void Initialize(ASTContext &context) override { this->context = &context; }
  bool HandleTopLevelDecl(DeclGroupRef DG) override {
    handleDeclGroup(DG);
    return true;
  }
};

class ExportFFIAction : public PluginASTAction {
  std::unique_ptr<llvm::raw_ostream> cs;
  std::unique_ptr<llvm::raw_ostream> cpp;

protected:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
    return std::make_unique<ExportFFIConsumer>(std::move(cs), std::move(cpp),
                                               CI);
  }
  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    auto &Diag = CI.getDiagnostics();
    if (args.size() == 2) {
      std::error_code ec;
      cs = std::make_unique<llvm::raw_fd_stream>(args[0], ec);
      if (ec) {
        unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                               "can not open output file '%0'");
        Diag.Report(DiagID) << args[0];
        return false;
      }
      cpp = std::make_unique<llvm::raw_fd_stream>(args[1], ec);
      if (ec) {
        unsigned DiagID = Diag.getCustomDiagID(DiagnosticsEngine::Error,
                                               "can not open output file '%0'");
        Diag.Report(DiagID) << args[0];
        return false;
      }
      return true;

    } else {
      unsigned DiagID = Diag.getCustomDiagID(
          DiagnosticsEngine::Error,
          "should provide two args as output chez-scheme file and C stub file");
      Diag.Report(DiagID);
      return false;
    }
  }
};
} // namespace MLIR_CAPI_ChezSchemeFFI
} // namespace

static FrontendPluginRegistry::Add<MLIR_CAPI_ChezSchemeFFI::ExportFFIAction>
    X("mlir-scheme-ffi", "export MLIR CAPI function/types with chez-schme ffi");
