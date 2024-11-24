// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::Dialect_::PDL::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;
use mlir_ffi_rs::common::mlirTypeIsNull;

// CHECK-LABEL: testAttributeType
fn testAttributeType(ctx: MlirContext) {
    unsafe {
        eprintln!("testAttributeType");

        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!pdl.attribute\0".as_ptr() as *const i8),
        );
        let constructedType = mlirPDLAttributeTypeGet(ctx);

        assert!(
            !mlirTypeIsNull(parsedType),
            "couldn't parse PDLAttributeType"
        );
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct PDLAttributeType"
        );

        // CHECK: parsedType isa PDLType: 1
        eprintln!("parsedType isa PDLType: {}", mlirTypeIsAPDLType(parsedType));
        // CHECK: parsedType isa PDLAttributeType: 1
        eprintln!(
            "parsedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprintln!(
            "parsedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprintln!(
            "parsedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprintln!(
            "parsedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprintln!(
            "parsedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprintln!(
            "constructedType isa PDLType: {}",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 1
        eprintln!(
            "constructedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprintln!(
            "constructedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprintln!(
            "constructedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprintln!(
            "constructedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprintln!(
            "constructedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: !pdl.attribute
        mlirTypeDump(parsedType);
        // CHECK: !pdl.attribute
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
            mlirStringRefCreateFromCString("!pdl.operation\0".as_ptr() as *const i8),
        );
        let constructedType = mlirPDLOperationTypeGet(ctx);

        assert!(
            !mlirTypeIsNull(parsedType),
            "couldn't parse PDLAttributeType"
        );
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct PDLAttributeType"
        );

        // CHECK: parsedType isa PDLType: 1
        eprintln!("parsedType isa PDLType: {}", mlirTypeIsAPDLType(parsedType));
        // CHECK: parsedType isa PDLAttributeType: 0
        eprintln!(
            "parsedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 1
        eprintln!(
            "parsedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprintln!(
            "parsedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprintln!(
            "parsedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprintln!(
            "parsedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprintln!(
            "constructedType isa PDLType: {}",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprintln!(
            "constructedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 1
        eprintln!(
            "constructedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprintln!(
            "constructedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprintln!(
            "constructedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprintln!(
            "constructedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: !pdl.operation
        mlirTypeDump(parsedType);
        // CHECK: !pdl.operation
        mlirTypeDump(constructedType);

        eprint!("\n\n");
    }
}

// CHECK-LABEL: testRangeType
fn testRangeType(ctx: MlirContext) {
    unsafe {
        eprintln!("testRangeType");

        let typeType = mlirPDLTypeTypeGet(ctx);
        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!pdl.range<type>\0".as_ptr() as *const i8),
        );
        let constructedType = mlirPDLRangeTypeGet(typeType);
        let elementType = mlirPDLRangeTypeGetElementType(constructedType);

        assert!(!mlirTypeIsNull(typeType), "couldn't get PDLTypeType");
        assert!(
            !mlirTypeIsNull(parsedType),
            "couldn't parse PDLAttributeType"
        );
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct PDLAttributeType"
        );

        // CHECK: parsedType isa PDLType: 1
        eprintln!("parsedType isa PDLType: {}", mlirTypeIsAPDLType(parsedType));
        // CHECK: parsedType isa PDLAttributeType: 0
        eprintln!(
            "parsedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprintln!(
            "parsedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 1
        eprintln!(
            "parsedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprintln!(
            "parsedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprintln!(
            "parsedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprintln!(
            "constructedType isa PDLType: {}",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprintln!(
            "constructedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprintln!(
            "constructedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 1
        eprintln!(
            "constructedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprintln!(
            "constructedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprintln!(
            "constructedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));
        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(typeType, elementType));

        // CHECK: !pdl.range<type>
        mlirTypeDump(parsedType);
        // CHECK: !pdl.range<type>
        mlirTypeDump(constructedType);
        // CHECK: !pdl.type
        mlirTypeDump(elementType);

        eprint!("\n\n");
    }
}

// CHECK-LABEL: testTypeType
fn testTypeType(ctx: MlirContext) {
    unsafe {
        eprintln!("testTypeType");

        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!pdl.type\0".as_ptr() as *const i8),
        );
        let constructedType = mlirPDLTypeTypeGet(ctx);

        assert!(
            !mlirTypeIsNull(parsedType),
            "couldn't parse PDLAttributeType"
        );
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct PDLAttributeType"
        );

        // CHECK: parsedType isa PDLType: 1
        eprintln!("parsedType isa PDLType: {}", mlirTypeIsAPDLType(parsedType));
        // CHECK: parsedType isa PDLAttributeType: 0
        eprintln!(
            "parsedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprintln!(
            "parsedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprintln!(
            "parsedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 1
        eprintln!(
            "parsedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprintln!(
            "parsedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprintln!(
            "constructedType isa PDLType: {}",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprintln!(
            "constructedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprintln!(
            "constructedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprintln!(
            "constructedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 1
        eprintln!(
            "constructedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprintln!(
            "constructedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: !pdl.type
        mlirTypeDump(parsedType);
        // CHECK: !pdl.type
        mlirTypeDump(constructedType);

        eprint!("\n\n");
    }
}

// CHECK-LABEL: testValueType
fn testValueType(ctx: MlirContext) {
    unsafe {
        eprintln!("testValueType");

        let parsedType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!pdl.value\0".as_ptr() as *const i8),
        );
        let constructedType = mlirPDLValueTypeGet(ctx);

        assert!(
            !mlirTypeIsNull(parsedType),
            "couldn't parse PDLAttributeType"
        );
        assert!(
            !mlirTypeIsNull(constructedType),
            "couldn't construct PDLAttributeType"
        );

        // CHECK: parsedType isa PDLType: 1
        eprintln!("parsedType isa PDLType: {}", mlirTypeIsAPDLType(parsedType));
        // CHECK: parsedType isa PDLAttributeType: 0
        eprintln!(
            "parsedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprintln!(
            "parsedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprintln!(
            "parsedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprintln!(
            "parsedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 1
        eprintln!(
            "parsedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprintln!(
            "constructedType isa PDLType: {}",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprintln!(
            "constructedType isa PDLAttributeType: {}",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprintln!(
            "constructedType isa PDLOperationType: {}",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprintln!(
            "constructedType isa PDLRangeType: {}",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprintln!(
            "constructedType isa PDLTypeType: {}",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 1
        eprintln!(
            "constructedType isa PDLValueType: {}",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(parsedType, constructedType));

        // CHECK: !pdl.value
        mlirTypeDump(parsedType);
        // CHECK: !pdl.value
        mlirTypeDump(constructedType);

        eprint!("\n\n");
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__pdl__(), ctx);
        testAttributeType(ctx);
        testOperationType(ctx);
        testRangeType(ctx);
        testTypeType(ctx);
        testValueType(ctx);
        mlirContextDestroy(ctx);
    }
}
