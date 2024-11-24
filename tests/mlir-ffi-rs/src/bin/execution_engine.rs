// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::Conversion::*;
use mlir_capi::ExecutionEngine::*;
use mlir_capi::Pass::*;
use mlir_capi::RegisterEverything::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

use mlir_ffi_rs::c_str_ptr;
use mlir_ffi_rs::common::{mlirExecutionEngineIsNull, mlirLogicalResultIsFailure};

fn registerAllUpstreamDialects(ctx: MlirContext) {
    unsafe {
        let registry = mlirDialectRegistryCreate();
        mlirRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);
    }
}

fn lowerModuleToLLVM(ctx: MlirContext, module: MlirModule) {
    unsafe {
        let pm = mlirPassManagerCreate(ctx);
        let opm = mlirPassManagerGetNestedUnder(
            pm,
            mlirStringRefCreateFromCString(c_str_ptr!("func.func")),
        );
        mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
        mlirOpPassManagerAddOwnedPass(opm, mlirCreateConversionArithToLLVMConversionPass());
        let status = mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(module));
        if mlirLogicalResultIsFailure(status) {
            eprintln!("Unexpected failure running pass pipeline");
            std::process::exit(2);
        }
        mlirPassManagerDestroy(pm);
    }
}

// CHECK-LABEL: Running test 'testSimpleExecution'
fn testSimpleExecution() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        let module = mlirModuleCreateParse(
            ctx,
            mlirStringRefCreateFromCString(
                // clang-format off
                c_str_ptr!(
                    "module {
  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
    %res = arith.addi %arg0, %arg0 : i32
    return %res : i32
  }
}"
                ),
            ),
        );
        // clang-format on
        lowerModuleToLLVM(ctx, module);
        mlirRegisterAllLLVMTranslations(ctx);
        let jit = mlirExecutionEngineCreate(
            module,
            /*optLevel=*/ 2,
            /*numPaths=*/ 0,
            /*sharedLibPaths=*/ std::ptr::null_mut(),
            /*enableObjectDump=*/ 0,
        );
        if mlirExecutionEngineIsNull(jit) {
            eprint!("Execution engine creation failed");
            std::process::exit(2);
        }
        let mut input = 42;
        let mut result = -1;
        let mut args = [
            &mut input as *mut _ as *mut std::ffi::c_void,
            &mut result as *mut _ as *mut std::ffi::c_void,
        ];
        if mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
            jit,
            mlirStringRefCreateFromCString(c_str_ptr!("add")),
            args.as_mut_ptr(),
        )) {
            eprint!("Execution engine creation failed");
            libc::abort();
        }
        // CHECK: Input: 42 Result: 84
        println!("Input: {} Result: {}", input, result);
        mlirExecutionEngineDestroy(jit);
        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
    }
}

// CHECK-LABEL: Running test 'testOmpCreation'
fn testOmpCreation() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        let module = mlirModuleCreateParse(
            ctx,
            mlirStringRefCreateFromCString(
                // clang-format off
                c_str_ptr!(
                    "module {
  func.func @main() attributes { llvm.emit_c_interface } {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %2 = arith.constant 2 : i32
    omp.parallel {
      omp.wsloop {
        omp.loop_nest (%3) : i32 = (%0) to (%2) step (%1) {
          omp.yield
        }
      }
      omp.terminator
    }
    llvm.return
  }
}\n"
                ),
            ),
        );
        // clang-format on
        lowerModuleToLLVM(ctx, module);

        // At this point all operations in the MLIR module have been lowered to the
        // 'llvm' dialect except 'omp' operations. The goal of this test is
        // guaranteeing that the execution engine C binding has registered OpenMP
        // translations and therefore does not fail when it encounters 'omp' ops.
        // We don't attempt to run the engine, since that would force us to link
        // against the OpenMP library.
        let jit = mlirExecutionEngineCreate(
            module,
            /*optLevel=*/ 2,
            /*numPaths=*/ 0,
            /*sharedLibPaths=*/ std::ptr::null_mut(),
            /*enableObjectDump=*/ 0,
        );
        if mlirExecutionEngineIsNull(jit) {
            eprint!("Engine creation failed with OpenMP");
            std::process::exit(2);
        }
        // CHECK: Engine creation succeeded with OpenMP
        println!("Engine creation succeeded with OpenMP");
        mlirExecutionEngineDestroy(jit);
        mlirModuleDestroy(module);
        mlirContextDestroy(ctx);
    }
}

fn main() {
    println!("Running test 'testSimpleExecution'");
    testSimpleExecution();
    println!("Running test 'testOmpCreation'");
    testOmpCreation();
}
