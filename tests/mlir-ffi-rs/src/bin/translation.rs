// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::Dialect_::LLVM::*;
use mlir_capi::RegisterEverything::*;
use mlir_capi::Support::*;
use mlir_capi::Target_::LLVMIR::*;
use mlir_capi::IR::*;

use llvm_capi::Core::*;

// CHECK-LABEL: testToLLVMIR()
fn testToLLVMIR(ctx: MlirContext) {
    unsafe {
        eprintln!("testToLLVMIR()");
        let llvmCtx = LLVMContextCreate();

        let moduleString = "llvm.func @add(%arg0: i64, %arg1: i64) -> i64 {
                                %0 = llvm.add %arg0, %arg1  : i64
                                llvm.return %0 : i64
                             }\0"
        .as_ptr() as *const i8;

        mlirRegisterAllLLVMTranslations(ctx);

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));

        let operation = mlirModuleGetOperation(module);

        let llvmModule = mlirTranslateModuleToLLVMIR(operation, llvmCtx);

        // clang-format off
        // CHECK: define i64 @add(i64 %[[arg1:.*]], i64 %[[arg2:.*]]) {
        // CHECK-NEXT:   %[[arg3:.*]] = add i64 %[[arg1]], %[[arg2]]
        // CHECK-NEXT:   ret i64 %[[arg3]]
        // CHECK-NEXT: }
        // clang-format on
        LLVMDumpModule(llvmModule);

        LLVMDisposeModule(llvmModule);
        mlirModuleDestroy(module);
        LLVMContextDispose(llvmCtx);
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("llvm\0".as_ptr() as *const i8),
        );
        testToLLVMIR(ctx);
        mlirContextDestroy(ctx);
    }
}
