// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::Dialect_::IRDL::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

fn main() {
    let irdlDialect = "
  irdl.dialect @foo {
    irdl.operation @op {
      %i32 = irdl.is i32
      irdl.results(%i32)
    }
  }
  irdl.dialect @bar {
    irdl.operation @op {
      %i32 = irdl.is i32
      irdl.operands(%i32)
    }
  }\0"
    .as_ptr() as *const i8;

    // CHECK:      module {
    // CHECK-NEXT:   %[[RES:.*]] = "foo.op"() : () -> i32
    // CHECK-NEXT:   "bar.op"(%[[RES]]) :  (i32) -> ()
    // CHECK-NEXT: }
    let newDialectUsage = "
  module {
    %res = \"foo.op\"() : () -> i32
    \"bar.op\"(%res) : (i32) -> ()
  }\0"
    .as_ptr() as *const i8;

    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleLoadDialect(mlirGetDialectHandle__irdl__(), ctx);

        let dialectDecl = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(irdlDialect));

        mlirLoadIRDLDialects(dialectDecl);
        mlirModuleDestroy(dialectDecl);

        let usingModule =
            mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(newDialectUsage));

        mlirOperationDump(mlirModuleGetOperation(usingModule));

        mlirModuleDestroy(usingModule);
        mlirContextDestroy(ctx);
    }
}
