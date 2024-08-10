#![allow(non_snake_case)]

use mlir::Dialect_::LLVM::*;
use mlir::RegisterEverything::*;
use mlir::Support::*;
use mlir::Target_::LLVMIR::*;
use mlir::IR::*;

use llvm::Core::*;

// CHECK-LABEL: testToLLVMIR()
fn testToLLVMIR(ctx: MlirContext) {
    unsafe {
        eprint!("testToLLVMIR()\n");
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
