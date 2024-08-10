#![allow(non_snake_case)]

use libc;

use mlir;
use mlir::Dialect_::Func::*;
use mlir::Pass::*;
use mlir::RegisterEverything::*;
use mlir::Support::*;
use mlir::Transforms::*;
use mlir::IR::*;
use mlir_ffi_rs::common::mlirLogicalResultIsSuccess;

fn registerAllUpstreamDialects(ctx: MlirContext) {
    unsafe {
        let registry = mlirDialectRegistryCreate();
        mlirRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);
    }
}

fn mlirLogicalResultIsFailure(res: MlirLogicalResult) -> bool {
    return res.value == 0;
}

fn mlirOperationIsNull(op: MlirOperation) -> bool {
    return op.ptr == std::ptr::null_mut();
}

fn mlirLogicalResultSuccess() -> MlirLogicalResult {
    MlirLogicalResult { value: 1 }
}
fn mlirLogicalResultFailure() -> MlirLogicalResult {
    MlirLogicalResult { value: 0 }
}

fn testRunPassOnModule() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        let funcAsm = "func.func @foo(%arg0 : i32) -> i32 {
  %res = arith.addi %arg0, %arg0 : i32
  return %res : i32 
}\0"
        .as_ptr() as *const i8;

        let func = mlirOperationCreateParse(
            ctx,
            mlirStringRefCreateFromCString(funcAsm),
            mlirStringRefCreateFromCString("funcAsm\0".as_ptr() as *const i8),
        );
        if func.ptr == std::ptr::null_mut() {
            eprint!("Unexpected failure parsing asm.\n");
            libc::exit(libc::EXIT_FAILURE);
        }

        // Run the print-op-stats pass on the top-level module:
        // CHECK-LABEL: Operations encountered:
        // CHECK: arith.addi        , 1
        // CHECK: func.func      , 1
        // CHECK: func.return        , 1
        {
            let pm = mlirPassManagerCreate(ctx);
            let printOpStatPass = mlirCreateTransformsPrintOpStats();
            mlirPassManagerAddOwnedPass(pm, printOpStatPass);
            let success = mlirPassManagerRunOnOp(pm, func);
            if mlirLogicalResultIsFailure(success) {
                eprint!("Unexpected failure running pass manager.\n");
                libc::exit(libc::EXIT_FAILURE);
            }
            mlirPassManagerDestroy(pm);
        }
        mlirOperationDestroy(func);
        mlirContextDestroy(ctx);
    }
}

fn testRunPassOnNestedModule() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        let moduleAsm = "module {
  func.func @foo(%arg0 : i32) -> i32 {
    %res = arith.addi %arg0, %arg0 : i32
    return %res : i32
  }
  module {
    func.func @bar(%arg0 : f32) -> f32 {
      %res = arith.addf %arg0, %arg0 : f32
      return %res : f32
    }
  }
}\0"
        .as_ptr() as *const i8;
        let module = mlirOperationCreateParse(
            ctx,
            mlirStringRefCreateFromCString(moduleAsm),
            mlirStringRefCreateFromCString("moduleAsm\0".as_ptr() as *const i8),
        );
        if mlirOperationIsNull(module) {
            libc::exit(1);
        }

        // Run the print-op-stats pass on functions under the top-level module:
        // CHECK-LABEL: Operations encountered:
        // CHECK: arith.addi        , 1
        // CHECK: func.func      , 1
        // CHECK: func.return        , 1
        {
            let pm = mlirPassManagerCreate(ctx);
            let nestedFuncPm = mlirPassManagerGetNestedUnder(
                pm,
                mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8),
            );
            let printOpStatPass = mlirCreateTransformsPrintOpStats();
            mlirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsFailure(success) {
                libc::exit(2);
            }
            mlirPassManagerDestroy(pm);
        }
        // Run the print-op-stats pass on functions under the nested module:
        // CHECK-LABEL: Operations encountered:
        // CHECK: arith.addf        , 1
        // CHECK: func.func      , 1
        // CHECK: func.return        , 1
        {
            let pm = mlirPassManagerCreate(ctx);
            let nestedModulePm = mlirPassManagerGetNestedUnder(
                pm,
                mlirStringRefCreateFromCString("builtin.module\0".as_ptr() as *const i8),
            );
            let nestedFuncPm = mlirOpPassManagerGetNestedUnder(
                nestedModulePm,
                mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8),
            );
            let printOpStatPass = mlirCreateTransformsPrintOpStats();
            mlirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsFailure(success) {
                libc::exit(2);
            }
            mlirPassManagerDestroy(pm);
        }

        mlirOperationDestroy(module);
        mlirContextDestroy(ctx);
    }
}

pub extern "C" fn printToStderr(r#str: MlirStringRef, _userData: *mut u8) {
    unsafe {
        let str_slice = std::slice::from_raw_parts(r#str.data as *const u8, r#str.length as usize);
        let str_str = std::str::from_utf8_unchecked(str_slice);
        eprint!("{}", str_str);
    }
}

fn testPrintPassPipeline() {
    unsafe {
        let ctx = mlirContextCreate();
        let pm = mlirPassManagerCreateOnOperation(
            ctx,
            mlirStringRefCreateFromCString("any\0".as_ptr() as *const i8),
        );
        // Populate the pass-manager
        let nestedModulePm = mlirPassManagerGetNestedUnder(
            pm,
            mlirStringRefCreateFromCString("builtin.module\0".as_ptr() as *const i8),
        );
        let nestedFuncPm = mlirOpPassManagerGetNestedUnder(
            nestedModulePm,
            mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8),
        );
        let printOpStatPass = mlirCreateTransformsPrintOpStats();
        mlirOpPassManagerAddOwnedPass(nestedFuncPm, printOpStatPass);

        // Print the top level pass manager
        //      CHECK: Top-level: any(
        // CHECK-SAME:   builtin.module(func.func(print-op-stats{json=false}))
        // CHECK-SAME: )
        eprint!("Top-level: ");
        mlirPrintPassPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        eprint!("\n");

        // Print the pipeline nested one level down
        // CHECK: Nested Module: builtin.module(func.func(print-op-stats{json=false}))
        eprint!("Nested Module: ");
        mlirPrintPassPipeline(nestedModulePm, printToStderr as _, std::ptr::null_mut());
        eprint!("\n");

        // Print the pipeline nested two levels down
        // CHECK: Nested Module>Func: func.func(print-op-stats{json=false})
        eprint!("Nested Module>Func: ");
        mlirPrintPassPipeline(nestedFuncPm, printToStderr as _, std::ptr::null_mut());
        eprint!("\n");

        mlirPassManagerDestroy(pm);
        mlirContextDestroy(ctx);
    }
}

fn testParsePassPipeline() {
    unsafe {
        let ctx = mlirContextCreate();
        let pm = mlirPassManagerCreate(ctx);
        // Try parse a pipeline.
        let mut status = mlirParsePassPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            mlirStringRefCreateFromCString(
                "builtin.module(func.func(print-op-stats{json=false}))\0".as_ptr() as *const i8,
            ),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        // Expect a failure, we haven't registered the print-op-stats pass yet.
        if mlirLogicalResultIsSuccess(status) {
            eprint!("Unexpected success parsing pipeline without registering the pass\n");
            libc::exit(libc::EXIT_FAILURE);
        }
        // Try again after registrating the pass.
        mlirRegisterTransformsPrintOpStats();
        status = mlirParsePassPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            mlirStringRefCreateFromCString(
                "builtin.module(func.func(print-op-stats{json=false}))\0".as_ptr() as *const i8,
            ),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        // Expect a failure, we haven't registered the print-op-stats pass yet.
        if mlirLogicalResultIsFailure(status) {
            eprint!("Unexpected failure parsing pipeline after registering the pass\n");
            libc::exit(libc::EXIT_FAILURE);
        }

        // CHECK: Round-trip: builtin.module(func.func(print-op-stats{json=false}))
        eprint!("Round-trip: ");
        mlirPrintPassPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        eprint!("\n");

        // Try appending a pass:
        status = mlirOpPassManagerAddPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            mlirStringRefCreateFromCString(
                "func.func(print-op-stats{json=false})\0".as_ptr() as *const i8
            ),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        if mlirLogicalResultIsFailure(status) {
            eprint!("Unexpected failure appending pipeline\n");
            libc::exit(libc::EXIT_FAILURE);
        }
        //      CHECK: Appended: builtin.module(
        // CHECK-SAME:   func.func(print-op-stats{json=false}),
        // CHECK-SAME:   func.func(print-op-stats{json=false})
        // CHECK-SAME: )
        eprint!("Appended: ");
        mlirPrintPassPipeline(
            mlirPassManagerGetAsOpPassManager(pm),
            printToStderr as _,
            std::ptr::null_mut(),
        );
        eprint!("\n");

        mlirPassManagerDestroy(pm);
        mlirContextDestroy(ctx);
    }
}

pub extern "C" fn dontPrint(_: MlirStringRef, _: *mut u8) {}

fn testParseErrorCapture() {
    unsafe {
        // CHECK-LABEL: testParseErrorCapture:
        eprint!("\nTEST: testParseErrorCapture:\n");

        let ctx = mlirContextCreate();
        let pm = mlirPassManagerCreate(ctx);
        let opm = mlirPassManagerGetAsOpPassManager(pm);
        let invalidPipeline = mlirStringRefCreateFromCString("invalid\0".as_ptr() as *const i8);

        // CHECK: mlirParsePassPipeline:
        // CHECK: expected pass pipeline to be wrapped with the anchor operation type
        eprint!("mlirParsePassPipeline:\n");
        if mlirLogicalResultIsSuccess(mlirParsePassPipeline(
            opm,
            invalidPipeline,
            printToStderr as _,
            std::ptr::null_mut(),
        )) {
            libc::exit(libc::EXIT_FAILURE);
        }
        eprint!("\n");

        // CHECK: mlirOpPassManagerAddPipeline:
        // CHECK: 'invalid' does not refer to a registered pass or pass pipeline
        eprint!("mlirOpPassManagerAddPipeline:\n");
        if mlirLogicalResultIsSuccess(mlirOpPassManagerAddPipeline(
            opm,
            invalidPipeline,
            printToStderr as _,
            std::ptr::null_mut(),
        )) {
            libc::exit(libc::EXIT_FAILURE);
        }
        eprint!("\n");

        // Make sure all output is going through the callback.
        // CHECK: dontPrint: <>
        eprint!("dontPrint: <");
        if mlirLogicalResultIsSuccess(mlirParsePassPipeline(
            opm,
            invalidPipeline,
            dontPrint as _,
            std::ptr::null_mut(),
        )) {
            libc::exit(libc::EXIT_FAILURE);
        }
        if mlirLogicalResultIsSuccess(mlirOpPassManagerAddPipeline(
            opm,
            invalidPipeline,
            dontPrint as _,
            std::ptr::null_mut(),
        )) {
            libc::exit(libc::EXIT_FAILURE);
        }
        eprint!(">\n");

        mlirPassManagerDestroy(pm);
        mlirContextDestroy(ctx);
    }
}

struct TestExternalPassUserData {
    pub constructCallCount: std::ffi::c_int,
    pub destructCallCount: std::ffi::c_int,
    pub initializeCallCount: std::ffi::c_int,
    pub cloneCallCount: std::ffi::c_int,
    pub runCallCount: std::ffi::c_int,
}

pub extern "C" fn testRunExternalPass(
    _op: MlirOperation,
    _pass: MlirExternalPass,
    userData: *mut u8,
) {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).runCallCount += 1;
    }
}

pub extern "C" fn testRunExternalFuncPass(
    op: MlirOperation,
    pass: MlirExternalPass,
    userData: *mut u8,
) {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).runCallCount += 1;
        let opName = mlirIdentifierStr(mlirOperationGetName(op));
        if 0 == mlirStringRefEqual(
            opName,
            mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8),
        ) {
            mlirExternalPassSignalFailure(pass);
        }
    }
}

pub extern "C" fn testInitializeExternalPass(
    _ctx: MlirContext,
    userData: *const u8,
) -> MlirLogicalResult {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).initializeCallCount += 1;
        mlirLogicalResultSuccess()
    }
}

pub extern "C" fn testInitializeFailingExternalPass(
    _ctx: MlirContext,
    userData: *const u8,
) -> MlirLogicalResult {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).initializeCallCount += 1;
        mlirLogicalResultFailure()
    }
}

pub extern "C" fn testRunFailingExternalPass(
    _op: MlirOperation,
    pass: MlirExternalPass,
    userData: *mut u8,
) {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).runCallCount += 1;
        mlirExternalPassSignalFailure(pass);
    }
}

pub extern "C" fn testConstructExternalPass(userData: *mut u8) {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).constructCallCount += 1;
    }
}

pub extern "C" fn testDestructExternalPass(userData: *mut u8) {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).destructCallCount += 1;
    }
}

pub extern "C" fn testCloneExternalPass(userData: *mut u8) -> *mut u8 {
    unsafe {
        (*(userData as *mut TestExternalPassUserData)).cloneCallCount += 1;
        userData
    }
}

fn makeTestExternalPassCallbacks(
    initializePass: *mut fn(ctx: MlirContext, userData: *mut u8) -> MlirLogicalResult,
    runPass: *mut fn(op: MlirOperation, MlirExternalPass, userData: *mut u8),
) -> MlirExternalPassCallbacks {
    return MlirExternalPassCallbacks {
        construct: testConstructExternalPass as _,
        destruct: testDestructExternalPass as _,
        initialize: initializePass as _,
        clone: testCloneExternalPass as _,
        run: runPass as _,
    };
}

fn testExternalPass() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        let moduleAsm = "module {                                 
  func.func @foo(%arg0 : i32) -> i32 {   
    %res = arith.addi %arg0, %arg0 : i32 
    return %res : i32                    
  }                                      
}\0"
        .as_ptr() as *const i8;
        let module = mlirOperationCreateParse(
            ctx,
            mlirStringRefCreateFromCString(moduleAsm),
            mlirStringRefCreateFromCString("moduleAsm\0".as_ptr() as *const i8),
        );
        if mlirOperationIsNull(module) {
            eprint!("Unexpected failure parsing module.\n");
            libc::exit(libc::EXIT_FAILURE);
        }

        let description = mlirStringRefCreateFromCString("\0".as_ptr() as *const i8);
        let emptyOpName = mlirStringRefCreateFromCString("\0".as_ptr() as *const i8);

        let typeIDAllocator = mlirTypeIDAllocatorCreate();

        // Run a generic pass
        {
            let passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            let name = mlirStringRefCreateFromCString("TestExternalPass\0".as_ptr() as *const i8);
            let argument =
                mlirStringRefCreateFromCString("test-external-pass\0".as_ptr() as *const i8);
            let mut userData = TestExternalPassUserData {
                constructCallCount: 0,
                destructCallCount: 0,
                initializeCallCount: 0,
                cloneCallCount: 0,
                runCallCount: 0,
            };

            let externalPass = mlirCreateExternalPass(
                passID,
                name,
                argument,
                description,
                emptyOpName,
                0,
                std::ptr::null_mut(),
                makeTestExternalPassCallbacks(std::ptr::null_mut(), testRunExternalPass as _),
                &mut userData as *mut _ as _,
            );

            if userData.constructCallCount != 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            let pm = mlirPassManagerCreate(ctx);
            mlirPassManagerAddOwnedPass(pm, externalPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsFailure(success) {
                eprint!("Unexpected failure running external pass.\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.runCallCount != 1 {
                eprint!("Expected runCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            mlirPassManagerDestroy(pm);

            if userData.destructCallCount != userData.constructCallCount {
                eprint!("Expected destructCallCount to be equal to constructCallCount\n");
                libc::exit(libc::EXIT_FAILURE);
            }
        }

        // Run a func operation pass
        {
            let passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            let name =
                mlirStringRefCreateFromCString("TestExternalFuncPass\0".as_ptr() as *const i8);
            let argument =
                mlirStringRefCreateFromCString("test-external-func-pass\0".as_ptr() as *const i8);
            let mut userData = TestExternalPassUserData {
                constructCallCount: 0,
                destructCallCount: 0,
                initializeCallCount: 0,
                cloneCallCount: 0,
                runCallCount: 0,
            };
            let mut funcHandle = mlirGetDialectHandle__func__();
            let funcOpName = mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8);

            let externalPass = mlirCreateExternalPass(
                passID,
                name,
                argument,
                description,
                funcOpName,
                1,
                &mut funcHandle,
                makeTestExternalPassCallbacks(std::ptr::null_mut(), testRunExternalFuncPass as _),
                &mut userData as *mut _ as _,
            );

            if userData.constructCallCount != 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            let pm = mlirPassManagerCreate(ctx);
            let nestedFuncPm = mlirPassManagerGetNestedUnder(pm, funcOpName);
            mlirOpPassManagerAddOwnedPass(nestedFuncPm, externalPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsFailure(success) {
                eprint!("Unexpected failure running external operation pass.\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            // Since this is a nested pass, it can be cloned and run in parallel
            if userData.cloneCallCount != userData.constructCallCount - 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            // The pass should only be run once this there is only one func op
            if userData.runCallCount != 1 {
                eprint!("Expected runCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            mlirPassManagerDestroy(pm);

            if userData.destructCallCount != userData.constructCallCount {
                eprint!("Expected destructCallCount to be equal to constructCallCount\n");
                libc::exit(libc::EXIT_FAILURE);
            }
        }

        // Run a pass with `initialize` set
        {
            let passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            let name = mlirStringRefCreateFromCString("TestExternalPass\0".as_ptr() as *const i8);
            let argument =
                mlirStringRefCreateFromCString("test-external-pass\0".as_ptr() as *const i8);
            let mut userData = TestExternalPassUserData {
                constructCallCount: 0,
                destructCallCount: 0,
                initializeCallCount: 0,
                cloneCallCount: 0,
                runCallCount: 0,
            };

            let externalPass = mlirCreateExternalPass(
                passID,
                name,
                argument,
                description,
                emptyOpName,
                0,
                std::ptr::null_mut(),
                makeTestExternalPassCallbacks(
                    testInitializeExternalPass as _,
                    testRunExternalPass as _,
                ),
                &mut userData as *mut _ as _,
            );

            if userData.constructCallCount != 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            let pm = mlirPassManagerCreate(ctx);
            mlirPassManagerAddOwnedPass(pm, externalPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsFailure(success) {
                eprint!("Unexpected failure running external pass.\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.initializeCallCount != 1 {
                eprint!("Expected initializeCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.runCallCount != 1 {
                eprint!("Expected runCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            mlirPassManagerDestroy(pm);

            if userData.destructCallCount != userData.constructCallCount {
                eprint!("Expected destructCallCount to be equal to constructCallCount\n");
                libc::exit(libc::EXIT_FAILURE);
            }
        }

        // Run a pass that fails during `initialize`
        {
            let passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            let name =
                mlirStringRefCreateFromCString("TestExternalFailingPass\0".as_ptr() as *const i8);
            let argument = mlirStringRefCreateFromCString(
                "test-external-failing-pass\0".as_ptr() as *const i8
            );
            let mut userData = TestExternalPassUserData {
                constructCallCount: 0,
                destructCallCount: 0,
                initializeCallCount: 0,
                cloneCallCount: 0,
                runCallCount: 0,
            };

            let externalPass = mlirCreateExternalPass(
                passID,
                name,
                argument,
                description,
                emptyOpName,
                0,
                std::ptr::null_mut(),
                makeTestExternalPassCallbacks(
                    testInitializeFailingExternalPass as _,
                    testRunExternalPass as _,
                ),
                &mut userData as *mut _ as _,
            );

            if userData.constructCallCount != 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            let pm = mlirPassManagerCreate(ctx);
            mlirPassManagerAddOwnedPass(pm, externalPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsSuccess(success) {
                eprint!("Expected failure running pass manager on failing external pass.\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.initializeCallCount != 1 {
                eprint!("Expected initializeCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.runCallCount != 0 {
                eprint!("Expected runCallCount to be 0\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            mlirPassManagerDestroy(pm);

            if userData.destructCallCount != userData.constructCallCount {
                eprint!("Expected destructCallCount to be equal to constructCallCount\n");
                libc::exit(libc::EXIT_FAILURE);
            }
        }

        // Run a pass that fails during `run`
        {
            let passID = mlirTypeIDAllocatorAllocateTypeID(typeIDAllocator);
            let name =
                mlirStringRefCreateFromCString("TestExternalFailingPass\0".as_ptr() as *const i8);
            let argument = mlirStringRefCreateFromCString(
                "test-external-failing-pass\0".as_ptr() as *const i8
            );
            let mut userData = TestExternalPassUserData {
                constructCallCount: 0,
                destructCallCount: 0,
                initializeCallCount: 0,
                cloneCallCount: 0,
                runCallCount: 0,
            };

            let externalPass = mlirCreateExternalPass(
                passID,
                name,
                argument,
                description,
                emptyOpName,
                0,
                std::ptr::null_mut(),
                makeTestExternalPassCallbacks(
                    std::ptr::null_mut(),
                    testRunFailingExternalPass as _,
                ),
                &mut userData as *mut _ as _,
            );

            if userData.constructCallCount != 1 {
                eprint!("Expected constructCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            let pm = mlirPassManagerCreate(ctx);
            mlirPassManagerAddOwnedPass(pm, externalPass);
            let success = mlirPassManagerRunOnOp(pm, module);
            if mlirLogicalResultIsSuccess(success) {
                eprint!("Expected failure running pass manager on failing external pass-01.\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            if userData.runCallCount != 1 {
                eprint!("Expected runCallCount to be 1\n");
                libc::exit(libc::EXIT_FAILURE);
            }

            mlirPassManagerDestroy(pm);

            if userData.destructCallCount != userData.constructCallCount {
                eprint!("Expected destructCallCount to be equal to constructCallCount\n");
                libc::exit(libc::EXIT_FAILURE);
            }
        }

        mlirTypeIDAllocatorDestroy(typeIDAllocator);
        mlirOperationDestroy(module);
        mlirContextDestroy(ctx);
    }
}

fn main() {
    testRunPassOnModule();
    testRunPassOnNestedModule();
    testPrintPassPipeline();
    testParsePassPipeline();
    testParseErrorCapture();
    testExternalPass();
}
