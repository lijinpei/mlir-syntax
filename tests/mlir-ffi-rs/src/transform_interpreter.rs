#![allow(non_snake_case)]

use mlir::Dialect_::Transform::*;
use mlir::Dialect_::Transform_::Interpreter::*;
use mlir::Support::*;
use mlir::IR::*;

use mlir_ffi_rs::common::{mlirLogicalResultIsFailure, mlirOperationIsNull};

fn testApplyNamedSequence(ctx: MlirContext) -> i32 {
    unsafe {
        eprint!("testApplyNamedSequence\n",);

        let module = "module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.print %root { name = \"from interpreter\" }: 
!transform.any_op
    transform.yield
  }
}\0"
        .as_ptr() as *const i8;

        let moduleStringRef = mlirStringRefCreateFromCString(module);
        let nameStringRef = mlirStringRefCreateFromCString("inline-module\0".as_ptr() as *const i8);

        let root = mlirOperationCreateParse(ctx, moduleStringRef, nameStringRef);
        if mlirOperationIsNull(root) {
            return 1;
        }
        let body = mlirRegionGetFirstBlock(mlirOperationGetRegion(root, 0));
        let entry = mlirBlockGetFirstOperation(body);

        let options = mlirTransformOptionsCreate();
        mlirTransformOptionsEnableExpensiveChecks(options, 1);
        mlirTransformOptionsEnforceSingleTopLevelTransformOp(options, 1);

        let result = mlirTransformApplyNamedSequence(root, entry, root, options);
        mlirTransformOptionsDestroy(options);
        mlirOperationDestroy(root);
        if mlirLogicalResultIsFailure(result) {
            return 2;
        }
        return 0;
    }
}
// CHECK-LABEL: testApplyNamedSequence
// CHECK: from interpreter
// CHECK: transform.named_sequence @__transform_main
// CHECK:   transform.print %arg0
// CHECK:   transform.yield

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
        let result = testApplyNamedSequence(ctx);
        mlirContextDestroy(ctx);
        if result != 0 {
            std::process::exit(result);
        }
    }
}
