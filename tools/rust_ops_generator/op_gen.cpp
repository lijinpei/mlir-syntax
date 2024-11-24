#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

#include <string>
#include <unordered_set>

using namespace llvm;
using namespace mlir;

namespace {
cl::opt<std::string> yamlConfig("rust-config", cl::Required,
                                cl::desc("<YAML config file>"));
cl::opt<std::string> selectedDialect("dialect", cl::Required,
                                     cl::desc("The dialect to gen for"));
class RustOpsGenerator;
struct IndentGuard {
  IndentGuard(RustOpsGenerator &opGen);
  ~IndentGuard();
  RustOpsGenerator &opGen;
};

class RustOpsGenerator {
  std::unordered_set<std::string> allParams;
  friend class IndentGuard;
  const RecordKeeper &records;
  raw_ostream &os;
  const Record *argClass, *resClass, *propClass, *regionClass, *succClass;
  const Record *typeCons, *attrCons;
  const Record *attrClass;
  unsigned currIndent = 0;
  void incIndent() { currIndent += 1; }
  void decIndent() { currIndent -= 1; }
  IndentGuard guardIndent() { return IndentGuard(*this); }
  llvm::raw_ostream &indent() {
    for (unsigned i = 0; i < currIndent; ++i) {
      os << "  ";
    }
    return os;
  }
  // bool emitArgs(const Record *op) {
  //   const auto *args = op->getValueAsDag("arguments");
  //   for (unsigned i = 0, e = args->getNumArgs(); i < e; ++i) {
  //     auto *arg = dyn_cast<DefInit>(args->getArg(i));
  //     assert(arg);
  //     auto *argRec = arg->getDef();
  //     auto *argCons = argRec;
  //     if (argRec->isSubClassOf(argClass)) {
  //       argCons = argRec->getValueAsDef("constraint");
  //     } else if (argRec->isSubClassOf(resClass)) {
  //       // WA for EmitC_AssignOp
  //       argCons = argRec->getValueAsDef("constraint");
  //     }
  //     if (argCons->isSubClassOf(typeCons)) {
  //     } else if (argCons->isSubClassOf(attrCons)) {
  //       if (argCons->isSubClassOf(attrClass)) {
  //       } else {
  //       }
  //     } else if (argCons->isSubClassOf(propClass)) {
  //     } else {
  //       errs() << "unknown argument constraint: " << *arg << '\n'
  //              << *op << '\n';
  //       return false;
  //     }
  //   }
  //   return true;
  // }
  bool emitFieldAccessors(const tblgen::Operator &op) { return true; }
  bool emitOp(const tblgen::Operator &op) {
    os << "pub struct " << op.getCppClassName() << " {\n";
    {
      auto _ = guardIndent();
      indent() << "pub op: MlirOperation,\n";
    }
    os << "}\n\n";
    os << "impl " << op.getCppClassName() << " {\n";
    if (!emitFieldAccessors(op)) {
      return false;
    }
    {
      bool hasAttrArg = false; // not counting derived attr
      auto _ = guardIndent();
      // create
      // create_from_unwrapped
      indent() << "pub fn create(builder_: &IRBuilder, loc_: Location";
      if (!op.allResultTypesKnown()) {
        os << ", resTys_: &[Type]";
      }
      for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
        tblgen::Argument arg = op.getArg(i);
        if (auto *ty = dyn_cast<tblgen::NamedTypeConstraint *>(arg)) {
          os << ", " << ty->name << ": ";
          if (ty->isVariadicOfVariadic()) {
            os << "&[&[Value]]";
          } else if (ty->isVariadic()) {
            os << "&[Value]";
          } else {
            os << "Value";
          }
        } else if (auto *prop = dyn_cast<tblgen::NamedProperty *>(arg)) {
          os << ", " << prop->name << ": "
             << prop->prop.getInterfaceType().str();
          hasAttrArg = true;
        } else if (auto *attr = dyn_cast<tblgen::NamedAttribute *>(arg)) {
          os << ", " << attr->name << ": " << attr->attr.getStorageType();
          hasAttrArg = true;
        } else {
          assert(false);
        }
      }
      os << ") -> Self {\n";
      {
        auto _ = guardIndent();
        indent() << "let opNameSlice_ = c_str!(" << op.getOperationName()
                 << ");\n";
        indent() << "let opNameStrRef_ = MlirStringRef {data: "
                    "opNameSlice_.as_ptr(), length: opNameSlice_.len() };\n";
        indent() << "let mut opState_ = mlirOperationStateGet(opNameStrRef_, "
                    "loc_);\n";
        if (op.allResultTypesKnown()) {
          indent() << "mlirOperationStateEnableResultTypeInference(&mut "
                      "opState_);\n";
        } else {
          indent() << "mlirOperationStateAddResults(&mut opState_, "
                      "resTys_.len(), resTys_.as_ptr());\n";
        }
        for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
          tblgen::Argument arg = op.getArg(i);
          if (auto *ty = dyn_cast<tblgen::NamedTypeConstraint *>(arg)) {
            if (ty->isVariadicOfVariadic()) {
              indent() << "for x_ in " << ty->name << " {\n";
              {
                auto guard = guardIndent();
                indent() << "mlirOperationStateAddOperands(&mut opState_, "
                            "x_.len(), x_.as_ptr());\n";
              }
              indent() << "}\n";
            } else if (ty->isVariadic()) {
              indent() << "mlirOperationStateAddOperands(&mut opState_, "
                       << ty->name << ".len(), " << ty->name << ".as_ptr());\n";
            } else {
              indent() << "mlirOperationStateAddOperands(&mut opState_, 1, &"
                       << ty->name << ");\n";
            }
          }
        }
        if (hasAttrArg) {
          for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
            tblgen::Argument arg = op.getArg(i);
            if (auto *prop = dyn_cast<tblgen::NamedProperty *>(arg)) {
              indent() << "let " << prop->name << "NameSlice_ = c_str!(\""
                       << prop->name << "\");\n";
              indent() << "let " << prop->name << "NameStrRef_ = MlirStringRef("
                       << prop->name << "NameSlice_.len(), " << prop->name
                       << "NameSlice_.as_ptr());\n";
            } else if (auto *attr = dyn_cast<tblgen::NamedAttribute *>(arg)) {
              indent() << "let " << attr->name << "NameSlice_ = c_str!(\""
                       << attr->name << "\");\n";
              indent() << "let " << attr->name << "NameStrRef_ = MlirStringRef("
                       << attr->name << "NameSlice_.len(), " << attr->name
                       << "NameSlice_.as_ptr());\n";
            }
          }
          indent() << "let attrs_: &[MlirNamedAttribute] = [";
          for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
            tblgen::Argument arg = op.getArg(i);
            if (auto *prop = dyn_cast<tblgen::NamedProperty *>(arg)) {
              os << "MlirNamedAttribute {name: " << prop->name
                 << "NameStrRef_, attribute: " << prop->name
                 << ".wrap_attr() },";
            } else if (auto *attr = dyn_cast<tblgen::NamedAttribute *>(arg)) {
              os << "MlirNamedAttribute {name: " << attr->name
                 << "NameStrRef_, attribute: " << attr->name
                 << ".wrap_attr() },";
            }
          }
          os << "];\n";
          indent() << "mlirOperationStateAddAttributes(&mut opState_, "
                      "attrs_.len(), attrs_.as_ptr());\n";
        }
        indent() << "let op_ = mlirOperationCreate(&mut opState_);\n";
        indent() << op.getCppClassName() << " {op: op_ }\n";
      }
      indent() << "}\n";
    }
    os << "}\n\n";

    /*
    const auto *results = op->getValueAsDag("results");
    const auto *regions = op->getValueAsDag("regions");
    const auto *successors = op->getValueAsDag("successors");
    os << op->getValueAsString("opName") << '\n';
    if (!emitArgs(op)) {
      return false;
    }
    for (unsigned i = 0, e = results->getNumArgs(); i < e; ++i) {
      auto *res = dyn_cast<DefInit>(results->getArg(i));
      assert(res);
      auto *resRec = res->getDef();
      auto *resCons = resRec;
      if (resRec->isSubClassOf(argClass)) {
        // WA for GPU_DynamicSharedMemoryOp
        resCons = resRec->getValueAsDef("constraint");
      } else if (resRec->isSubClassOf(resClass)) {
        resCons = resRec->getValueAsDef("constraint");
      }
      if (resCons->isSubClassOf(typeCons)) {
      } else {
        errs() << "unknown result constraint: " << *res << '\n' << *op << '\n';
        return false;
      }
    }
    for (unsigned i = 0, e = regions->getNumArgs(); i < e; ++i) {
      auto *region = dyn_cast<DefInit>(regions->getArg(i));
      assert(region);
      auto *regionRec = region->getDef();
      if (regionRec->isSubClassOf(regionClass)) {
      } else {
        errs() << "unknown region constraint: " << *region << '\n'
               << *op << '\n';
        return false;
      }
    }
    for (unsigned i = 0, e = successors->getNumArgs(); i < e; ++i) {
      auto *succ = dyn_cast<DefInit>(successors->getArg(i));
      assert(succ);
      auto *succRec = succ->getDef();
      if (succRec->isSubClassOf(succClass)) {
      } else {
        errs() << "unknown successor constraint: " << *succ << '\n'
               << *op << '\n';
        return false;
      }
    }
    */
    return true;
  }

public:
  RustOpsGenerator(const RecordKeeper &records, raw_ostream &os)
      : records(records), os(os) {
    argClass = records.getClass("Arg");
    resClass = records.getClass("Res");
    propClass = records.getClass("Property");
    regionClass = records.getClass("Region");
    succClass = records.getClass("Successor");
    typeCons = records.getClass("TypeConstraint");
    attrCons = records.getClass("AttrConstraint");
    attrClass = records.getClass("Attr");
  }
  bool doIt() {
    auto dialects = records.getAllDerivedDefinitions("Dialect");
    auto dialectIt = llvm::find_if(dialects, [](const Record *def) {
      return def->getValueAsString("name") == selectedDialect;
    });
    if (dialectIt == dialects.end()) {
      llvm::errs() << "specified dialect: " << selectedDialect
                   << " not found\n";
      return false;
    }
    os << "#![allow(non_camel_case_types)]\n";
    os << "#![allow(unused_imports)]\n";
    os << "use mlir::IR::*;\n\n";
    auto ops = records.getAllDerivedDefinitions("Op");
    for (auto *op : ops) {
      if (!emitOp(tblgen::Operator(*op))) {
        return false;
      }
    }
    for (const auto &str : allParams) {
      llvm::StringRef strRef = str;
      strRef.consume_front("::mlir::");
      llvm::errs() << strRef << '\n';
    }
    return true;
  }
};

IndentGuard::IndentGuard(RustOpsGenerator &opGen) : opGen(opGen) {
  opGen.incIndent();
}
IndentGuard::~IndentGuard() { opGen.decIndent(); };

// Generator that prints ops in rust.
GenRegistration genRustOps("gen-rust-ops",
                           "Generate op definitions for use in rust",
                           [](const RecordKeeper &records, raw_ostream &os) {
                             return !RustOpsGenerator(records, os).doIt();
                           });

} // namespace
