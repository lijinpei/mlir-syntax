// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::Dialect_::Transform::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

use mlir_ffi_rs::common::mlirTypeIsNull;

// CHECK-LABEL: testAnyOpType
fn testAnyOpType(ctx: MlirContext) {
    unsafe {
        eprintln!("testAnyOpType");

        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!transform.any_op\0".as_ptr() as *const i8),
        );
        let constructedType = mlirTransformAnyOpTypeGet(ctx);

        assert!(!mlirTypeIsNull(parsedType), "couldn't parse AnyOpType");
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct AnyOpType"
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: parsedType isa AnyOpType: 1
        eprintln!(
            "parsedType isa AnyOpType: {}",
            mlirTypeIsATransformAnyOpType(parsedType)
        );
        // CHECK: parsedType isa OperationType: 0
        eprintln!(
            "parsedType isa OperationType: {}",
            mlirTypeIsATransformOperationType(parsedType)
        );

        // CHECK: !transform.any_op
        mlirTypeDump(constructedType);

        eprint!("\n\n");
    }
}

// CHECK-LABEL: testOperationType
fn testOperationType(ctx: MlirContext) {
    unsafe {
        eprintln!("testOperationType");

        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!transform.op<\"foo.bar\">\0".as_ptr() as *const i8),
        );
        let constructedType = mlirTransformOperationTypeGet(
            ctx,
            mlirStringRefCreateFromCString("foo.bar\0".as_ptr() as *const i8),
        );

        assert!(!mlirTypeIsNull(parsedType), "couldn't parse AnyOpType");
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct AnyOpType"
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: parsedType isa AnyOpType: 0
        eprintln!(
            "parsedType isa AnyOpType: {}",
            mlirTypeIsATransformAnyOpType(parsedType)
        );
        // CHECK: parsedType isa OperationType: 1
        eprintln!(
            "parsedType isa OperationType: {}",
            mlirTypeIsATransformOperationType(parsedType)
        );

        // CHECK: operation name equal: 1
        let operationName = mlirTransformOperationTypeGetOperationName(constructedType);
        eprintln!(
            "operation name equal: {}",
            mlirStringRefEqual(
                operationName,
                mlirStringRefCreateFromCString("foo.bar\0".as_ptr() as *const i8)
            )
        );

        // CHECK: !transform.op<"foo.bar">
        mlirTypeDump(constructedType);

        eprint!("\n\n");
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__transform__(), ctx);
        testAnyOpType(ctx);
        testOperationType(ctx);
        mlirContextDestroy(ctx);
    }
}
