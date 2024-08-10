#![allow(non_snake_case)]

use mlir;
use mlir::Dialect_::PDL::*;
use mlir::Support::*;
use mlir::IR::*;
use mlir_ffi_rs::common::mlirTypeIsNull;

// CHECK-LABEL: testAttributeType
fn testAttributeType(ctx: MlirContext) {
    unsafe {
        eprint!("testAttributeType\n");

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
        eprint!(
            "parsedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(parsedType)
        );
        // CHECK: parsedType isa PDLAttributeType: 1
        eprint!(
            "parsedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprint!(
            "parsedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprint!(
            "parsedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprint!(
            "parsedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprint!(
            "parsedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprint!(
            "constructedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 1
        eprint!(
            "constructedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprint!(
            "constructedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprint!(
            "constructedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprint!(
            "constructedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprint!(
            "constructedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(parsedType, constructedType));

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
        eprint!("testOperationType\n");

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
        eprint!(
            "parsedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(parsedType)
        );
        // CHECK: parsedType isa PDLAttributeType: 0
        eprint!(
            "parsedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 1
        eprint!(
            "parsedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprint!(
            "parsedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprint!(
            "parsedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprint!(
            "parsedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprint!(
            "constructedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprint!(
            "constructedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 1
        eprint!(
            "constructedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprint!(
            "constructedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprint!(
            "constructedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprint!(
            "constructedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(parsedType, constructedType));

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
        eprint!("testRangeType\n");

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
        eprint!(
            "parsedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(parsedType)
        );
        // CHECK: parsedType isa PDLAttributeType: 0
        eprint!(
            "parsedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprint!(
            "parsedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 1
        eprint!(
            "parsedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprint!(
            "parsedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprint!(
            "parsedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprint!(
            "constructedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprint!(
            "constructedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprint!(
            "constructedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 1
        eprint!(
            "constructedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprint!(
            "constructedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprint!(
            "constructedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(parsedType, constructedType));
        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(typeType, elementType));

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
        eprint!("testTypeType\n");

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
        eprint!(
            "parsedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(parsedType)
        );
        // CHECK: parsedType isa PDLAttributeType: 0
        eprint!(
            "parsedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprint!(
            "parsedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprint!(
            "parsedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 1
        eprint!(
            "parsedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 0
        eprint!(
            "parsedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprint!(
            "constructedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprint!(
            "constructedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprint!(
            "constructedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprint!(
            "constructedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 1
        eprint!(
            "constructedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 0
        eprint!(
            "constructedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(parsedType, constructedType));

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
        eprint!("testValueType\n");

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
        eprint!(
            "parsedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(parsedType)
        );
        // CHECK: parsedType isa PDLAttributeType: 0
        eprint!(
            "parsedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(parsedType)
        );
        // CHECK: parsedType isa PDLOperationType: 0
        eprint!(
            "parsedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(parsedType)
        );
        // CHECK: parsedType isa PDLRangeType: 0
        eprint!(
            "parsedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(parsedType)
        );
        // CHECK: parsedType isa PDLTypeType: 0
        eprint!(
            "parsedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(parsedType)
        );
        // CHECK: parsedType isa PDLValueType: 1
        eprint!(
            "parsedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(parsedType)
        );

        // CHECK: constructedType isa PDLType: 1
        eprint!(
            "constructedType isa PDLType: {}\n",
            mlirTypeIsAPDLType(constructedType)
        );
        // CHECK: constructedType isa PDLAttributeType: 0
        eprint!(
            "constructedType isa PDLAttributeType: {}\n",
            mlirTypeIsAPDLAttributeType(constructedType)
        );
        // CHECK: constructedType isa PDLOperationType: 0
        eprint!(
            "constructedType isa PDLOperationType: {}\n",
            mlirTypeIsAPDLOperationType(constructedType)
        );
        // CHECK: constructedType isa PDLRangeType: 0
        eprint!(
            "constructedType isa PDLRangeType: {}\n",
            mlirTypeIsAPDLRangeType(constructedType)
        );
        // CHECK: constructedType isa PDLTypeType: 0
        eprint!(
            "constructedType isa PDLTypeType: {}\n",
            mlirTypeIsAPDLTypeType(constructedType)
        );
        // CHECK: constructedType isa PDLValueType: 1
        eprint!(
            "constructedType isa PDLValueType: {}\n",
            mlirTypeIsAPDLValueType(constructedType)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(parsedType, constructedType));

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
