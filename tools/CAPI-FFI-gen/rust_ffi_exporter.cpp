#include "capi_ffi_generator.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <regex>
#include <vector>

// FIXME: types contained in C function prototype could be incomplete.
// Some non-ideal fix:
// 1. Add a hook for user to add extra use statement. (done)
// 2. discard function with incomplete param/ret type, and gives warning.

using namespace llvm;
using namespace clang;

namespace capi_ffi_gen {
namespace {
struct RustFFIExporter : FFIExporter {
  unsigned nextAnonNameIdx = 0;
  DenseMap<TagDecl *, std::string> anonTags;
  std::vector<std::string> thisCrateInclPaths;
  std::vector<FunctionDecl *> funcsToExport;
  void getThisCrateIncPaths(const YAML::Node &paths) {
    for (const auto &path : paths) {
      auto pathStr = path.as<std::string>();
      if (!StringRef(pathStr).ends_with('/')) {
        pathStr.push_back('/');
      }
      thisCrateInclPaths.push_back(pathStr);
    }
  }
  std::string getUseForRust(StringRef incFileName) {
    if (incFileName.ends_with(".rs")) {
      incFileName = incFileName.drop_back(3);
    }
    std::string res;
    while (true) {
      auto pos = incFileName.find('/');
      if (!res.empty()) {
        res += "::";
      }
      res += incFileName.substr(0, pos);
      if (pos == StringRef::npos) {
        break;
      }
      incFileName = incFileName.substr(pos + 1);
    }
    return res;
  }
  std::optional<std::string> getUseForIncl(StringRef incFileName) {
    if (incFileName.ends_with(".h")) {
      incFileName = incFileName.drop_back(2);
    } else {
      return std::nullopt;
    }
    for (const auto &path : thisCrateInclPaths) {
      if (!incFileName.starts_with(path)) {
        continue;
      }
      incFileName = incFileName.drop_front(path.size());
      std::string res;
      while (true) {
        auto [part1, part2] = incFileName.split('/');
        if (part2.empty()) {
          res += part1;
          break;
        } else {
          res += part1.substr(0, part1.size() - 1);
          res += "_::";
          incFileName = part2;
        }
      }
      return res;
    }
    return std::nullopt;
  }
  StringRef getOrCreateName(TagDecl *decl) {
    auto name = decl->getName();
    if (!name.empty()) {
      return name;
    }
    auto res = anonTags.try_emplace(decl, std::string());
    if (res.second) {
      res.first->second = "__anon_" + std::to_string(nextAnonNameIdx++);
    }
    return res.first->second;
  }
  enum TypeRefContext {
    TRC_Return,
    TRC_Pointee,
  };
  YAML::Node config;
  std::unique_ptr<raw_fd_ostream> fout_;
  std::string outputPath;
  raw_ostream &fout() {
    if (!hasError) {
      return *fout_;
    } else {
      return nulls();
    }
  }
  void newline() { fout() << '\n'; }
  void indent() {
    for (unsigned i = 0; i < indentLevel; ++i) {
      for (unsigned j = 0; j < NumIndent; ++j) {
        fout() << ' ';
      }
    }
  }

  RustFFIExporter(StringRef outputPath, const YAML::Node &config,
                  FFIExporterContext context)
      : config(config), FFIExporter("rust", context) {
    std::error_code ec;
    fout_ = std::make_unique<raw_fd_ostream>(outputPath, ec);
    if (ec) {
      diagError(Twine("can not open output file ") + outputPath);
      fout_.reset();
    }
    getThisCrateIncPaths(config["include_path"]);
    fout() << "// Don't Edit! Generated with capi-ffi-gen\n";
    indent();
    fout() << "#![allow(unused_imports)]\n";
    indent();
    fout() << "#![allow(non_camel_case_types)]\n";
    indent();
    fout() << "#![allow(non_upper_case_globals)]\n";
    indent();
    fout() << "#![allow(non_camel_case_types)]\n\n";
    fout() << "use std::convert::{From, Into};\n";
    fout() << "use std::marker::PhantomData;\n";
    auto extra_preamble = config["extra_preamble"];
    if (extra_preamble) {
      for (const auto &kv : extra_preamble) {
        if (kv.first.as<std::string>() == outputPath) {
          fout() << kv.second.as<std::string>();
        }
      }
    }
    // StringRef modName;
    // auto pathSepPos = outputPath.rfind('/');
    // if (pathSepPos == StringRef::npos) {
    //   modName = outputPath;
    // } else {
    //   modName = outputPath.substr(pathSepPos + 1);
    // }
    // modName.consume_back(".rs");
    // this->modName = modName.str();
    this->outputPath = outputPath.str();
  }
  Type *peelTypedef(const Type *ty) {
    while (true) {
      if (auto typedefTy = dyn_cast<TypedefType>(ty)) {
        ty = typedefTy->desugar().getTypePtr();
        continue;
      }
      if (auto elabTy = dyn_cast<ElaboratedType>(ty)) {
        ty = elabTy->desugar().getTypePtr();
        continue;
      }
      return const_cast<Type *>(ty);
    }
  }
  void emitTypeReference(raw_ostream &os, QualType qualType,
                         TypeRefContext trc = TRC_Return) {
    emitTypeReference(os, qualType.getTypePtrOrNull(), trc);
  }
  void emitTypeReference(raw_ostream &os, const Type *type,
                         TypeRefContext trc = TRC_Return) {
    if (auto *parenType = dyn_cast<ParenType>(type)) {
      return emitTypeReference(os, parenType->desugar(), trc);
    }
    if (auto *elabType = dyn_cast<ElaboratedType>(type)) {
      return emitTypeReference(os, elabType->desugar(), trc);
    }
    if (auto *decayType = dyn_cast<DecayedType>(type)) {
      return emitTypeReference(os, decayType->getDecayedType(), trc);
    }
    if (auto *recordType = dyn_cast<RecordType>(type)) {
      auto *recordDecl = recordType->getDecl();
      auto name = recordDecl->getName();
      if (name.empty()) {
        auto iter = anonTags.find(recordDecl);
        if (iter == anonTags.end()) {
          type->dump();
          recordDecl->dump();
        }
        assert(iter != anonTags.end());
        os << "Struct" << iter->second;
        return;
      }
      auto str = QualType(type, {}).getAsString();
      str = std::regex_replace(str, std::regex("^struct "), "");
      // FIXME: make this configurable, or at the end of TU, see if a record has
      // definition.
      if (str.find("Opaque") != std::string::npos || str == "LLVMTarget" ||
          str == "LLVMComdat") {
        os << "u8";
        return;
      }
      // FIXME: add a hook
      os << "Struct" << str;
      return;
    }
    if (auto *enumType = dyn_cast<EnumType>(type)) {
      auto *enumDecl = enumType->getDecl();
      auto name = enumDecl->getName();
      if (name.empty()) {
        emitTypeReference(os, enumDecl->getIntegerType());
        return;
      }
      auto str = QualType(type, {}).getAsString();
      // FIXME: add a hook
      str = std::regex_replace(str, std::regex("^enum "), "Enum");
      os << str;
      return;
    }
    if (auto *typedefType = dyn_cast<TypedefType>(type)) {
      auto *underlyigTy = peelTypedef(typedefType);
      if (isa<BuiltinType, EnumType>(underlyigTy)) {
        emitTypeReference(os, underlyigTy);
        return;
      }
      auto *typedefDecl = typedefType->getDecl();
      os << typedefDecl->getName();
      return;
    }
    if (auto *funcTy = dyn_cast<FunctionProtoType>(type)) {
      os << "extern fn (";
      bool isFirst = true;
      for (auto parTy : funcTy->getParamTypes()) {
        if (!isFirst) {
          os << ", ";
        }
        emitTypeReference(os, parTy);
        isFirst = false;
      }
      os << ") -> ";
      emitTypeReference(os, funcTy->getReturnType());
      return;
    }
    if (auto *ptrTy = dyn_cast<PointerType>(type)) {
      auto pointee = ptrTy->getPointeeType();
      if (pointee.isConstQualified()) {
        os << "*const ";
        pointee.removeLocalConst();
      } else {
        os << "*mut ";
      }
      emitTypeReference(os, pointee, TRC_Pointee);
      return;
    }
    if (auto arrayType = dyn_cast<ConstantArrayType>(type)) {
      os << '[';
      emitTypeReference(os, arrayType->getElementType());
      os << "; ";
      os << arrayType->getZExtSize();
      os << ']';
      return;
    }
    if (auto *builtinType = dyn_cast<BuiltinType>(type)) {
      if (type->isVoidType()) {
        if (trc == TRC_Pointee) {
          os << "std::ffi::c_void";
        } else {
          os << "()";
        }
        return;
      }
      auto kind = builtinType->getKind();
      if (kind == BuiltinType::Char_U || kind == BuiltinType::Char_S) {
        os << "std::ffi::c_char";
        return;
      }
      if (kind == BuiltinType::UChar) {
        os << "std::ffi::c_uchar";
        return;
      }
      if (kind == BuiltinType::SChar) {
        os << "std::ffi::c_schar";
        return;
      }
      if (kind == BuiltinType::Short) {
        os << "std::ffi::c_short";
        return;
      }
      if (kind == BuiltinType::UShort) {
        os << "std::ffi::c_ushort";
        return;
      }
      if (kind == BuiltinType::Int) {
        os << "std::ffi::c_int";
        return;
      }
      if (kind == BuiltinType::UInt) {
        os << "std::ffi::c_uint";
        return;
      }
      if (kind == BuiltinType::Long) {
        os << "std::ffi::c_long";
        return;
      }
      if (kind == BuiltinType::ULong) {
        os << "std::ffi::c_ulong";
        return;
      }
      if (kind == BuiltinType::LongLong) {
        os << "std::ffi::c_longlong";
        return;
      }
      if (kind == BuiltinType::ULongLong) {
        os << "std::ffi::c_ulonglong";
        return;
      }
      if (type->isIntegralType(getASTContext())) {
        if (type->isSignedIntegerType()) {
          os << "i";
          os << getASTContext().getTypeSizeInChars(type).getQuantity() * 8;
          return;
        }
        os << "u";
        os << getASTContext().getTypeSizeInChars(type).getQuantity() * 8;
        return;
      }
      if (builtinType->isFloatingPoint()) {
        os << "f";
        os << getASTContext().getTypeSizeInChars(type).getQuantity() * 8;
        return;
      }
    }
    {
      auto &Diag = getDiagnostics();
      unsigned DiagID = Diag.getCustomDiagID(
          DiagnosticsEngine::Error, "unhandled type in emitTypeReference: %0");
      std::string str;
      llvm::raw_string_ostream sos(str);
      type->dump(sos, getASTContext());
      Diag.Report(DiagID) << str;
    }
  }
  bool exportTypeDefNameDecl(clang::TypedefNameDecl *typedefDecl) override {
    indent();
    fout() << "pub type " << typedefDecl->getName() << " = ";
    emitTypeReference(fout(), typedefDecl->getUnderlyingType());
    fout() << ";\n";
    return true;
  }
  bool exportEnumDecl(EnumDecl *enumDecl) override {
    std::string intTyStr;
    raw_string_ostream ss(intTyStr);
    emitTypeReference(ss, enumDecl->getIntegerType());
    newline();
    for (auto *etor : enumDecl->enumerators()) {
      indent();
      fout() << "pub const " << etor->getName() << ": " << intTyStr << " = "
             << etor->getInitVal() << ";\n";
    }
    auto name = enumDecl->getName();
    if (!name.empty()) {
      indent();
      fout() << "pub type Enum" << name << " = " << intTyStr << ";\n";
    }

    return true;
  }
  bool exportRecordDecl(RecordDecl *recordDecl) override {
    newline();
    fout() << "#[repr(C)]\n";
    indent();
    auto name = getOrCreateName(recordDecl);
    fout() << "#[derive(Copy, Clone)]\n";
    indent();
    fout() << "pub struct Struct" << name << " {\n";
    {
      IncIndent incInd(*this);
      LangOptions refLangOpt;
      PrintingPolicy refPrintPolicy{refLangOpt};
      for (auto *fieldDecl : recordDecl->fields()) {
        indent();
        fout() << "pub ";
        fieldDecl->printName(fout(), refPrintPolicy);
        fout() << ": ";
        emitTypeReference(fout(), fieldDecl->getType());
        fout() << ",\n";
      }
    }
    indent();
    fout() << "}\n";
    return true;
  }
  bool exportFunctionDecl(FunctionDecl *funcDecl) override {
    funcsToExport.push_back(funcDecl);
    return true;
  }
  void exportFunctionDecl_(FunctionDecl *funcDecl) {
    newline();
    indent();
    fout() << "pub fn " << funcDecl->getName();
    fout() << '(';
    bool isFirst = true;
    for (auto *parDecl : funcDecl->parameters()) {
      if (!isFirst) {
        fout() << ", ";
      }
      auto name = parDecl->getName();
      if (name.empty()) {
        fout() << '_';
      } else {
        // TODO: extract an api
        if (name == "type") {
          name = "r#type";
        }
        if (name == "mod") {
          name = "r#mod";
        }
        if (name == "self") {
          name = "self_";
        }
        fout() << name;
      }
      fout() << ": ";
      emitTypeReference(fout(), parDecl->getType());
      isFirst = false;
    }
    fout() << ") -> ";
    emitTypeReference(fout(), funcDecl->getReturnType());
    fout() << ";\n";
  }
  void exportGenericFunctionImpl_(FunctionDecl *funcDecl,
                                  const std::string &usePath) {
    auto retTy = funcDecl->getReturnType();
    bool retVoid = retTy->isVoidType();
    bool noParam = funcDecl->parameters().empty();
    if (noParam && retVoid) {
      return;
    }
    std::string retTyName;
    if (!retVoid) {
      llvm::raw_string_ostream retTypeOS(retTyName);
      emitTypeReference(retTypeOS, retTy);
      fout() << "impl<Tret_> FFIVal_<Tret_> where Tret_: From<" << retTyName
             << "> {\n";
    } else {
      fout() << "impl FFIVoid_ {\n";
    }
    {
      IncIndent incInd(*this);
      std::string typeParamsStr, boundsStr, paramsStr, bodyStr;
      indent();
      fout() << "pub unsafe fn " << funcDecl->getName();
      if (!noParam) {
        llvm::raw_string_ostream typeParams(typeParamsStr);
        llvm::raw_string_ostream bounds(boundsStr);
        llvm::raw_string_ostream params(paramsStr);
        llvm::raw_string_ostream body(bodyStr);
        fout() << "<";
        for (auto [idx, parDecl] : llvm::enumerate(funcDecl->parameters())) {

          if (idx != 0) {
            params << ", ";
            body << ", ";
            bounds << ", ";
            typeParams << ", ";
          }
          std::string argName = parDecl->getName().str();
          if (argName.empty()) {
            argName = "arg" + std::to_string(idx);
          }
          argName += "_";
          std::string genericName = "T" + std::to_string(idx) + "_";
          std::string typeName;
          llvm::raw_string_ostream typeOS(typeName);
          emitTypeReference(typeOS, parDecl->getType());
          typeParams << genericName;
          bounds << " " << genericName << ": Into<" << typeName << ">";
          params << argName << ": " << " " << genericName;
          body << "Into::<" << typeName << ">::into(" << argName << ")";
        }
        fout() << typeParamsStr << ">";
      }
      fout() << "(" << paramsStr << ")";
      if (!retVoid) {
        fout() << "-> Tret_";
      }
      fout() << "\n";
      if (!noParam) {
        indent();
        fout() << "where\n";
        IncIndent incInd(*this);
        indent();
        fout() << boundsStr << "\n";
      }
      indent();
      fout() << "{\n";
      auto emitBody = [&]() {
        indent();
        fout() << "unsafe {\n";
        {
          IncIndent incInd(*this);
          indent();
          fout() << "crate::" << usePath << "::" << funcDecl->getName() << "("
                 << bodyStr << ")\n";
        }
        indent();
        fout() << "}\n";
      };
      {
        IncIndent incInd(*this);
        if (retVoid) {
          emitBody();
        } else {
          indent();
          fout() << "Into::<Tret_>::into(\n";
          {
            IncIndent incInd(*this);
            emitBody();
          }
          indent();
          fout() << ")\n";
        }
      }
      indent();
      fout() << "}\n";
    }
    fout() << "}\n\n";
  }
  void finishTranslationUnit(ASTContext &Ctx) override {
    fout() << "\n#[link(name = \"" << config["link_lib"].as<std::string>()
           << "\")]\n";
    fout() << "extern {\n";
    {
      IncIndent incInd(*this);
      for (auto *funcDecl : funcsToExport) {
        exportFunctionDecl_(funcDecl);
      }
      fout() << "\n";
    }
    fout() << "}\n\n";
    indent();
    fout() << "pub struct FFIVal_<Tret_> {e_: PhantomData<Tret_>,}\n";
    indent();
    fout() << "pub struct FFIVoid_;\n\n";
    auto usePath = getUseForRust(outputPath);
    // indent();
    //  fout() << "use crate::" << usePath << "::*;\n";
    for (auto *funcDecl : funcsToExport) {
      exportGenericFunctionImpl_(funcDecl, usePath);
    }
  }
  bool handleMainFileInclude(llvm::StringRef fileName) override {
    auto usePath = getUseForIncl(fileName);
    if (!usePath) {
      return true;
    }
    indent();
    fout() << "use crate::" << *usePath << "::*;\n";
    return true;
  }
};
} // namespace
std::unique_ptr<FFIExporter> createRustExporter(StringRef outputPath,
                                                const YAML::Node &config,
                                                FFIExporterContext context) {
  return std::make_unique<RustFFIExporter>(outputPath, config, context);
}
} // namespace capi_ffi_gen
