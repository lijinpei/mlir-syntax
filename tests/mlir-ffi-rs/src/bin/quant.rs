// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::BuiltinTypes::*;
use mlir_capi::Dialect_::Quant::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

use mlir_ffi_rs::common::mlirTypeIsNull;

// CHECK-LABEL: testTypeHierarchy
fn testTypeHierarchy(ctx: MlirContext) {
    unsafe {
        eprintln!("testTypeHierarchy");

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let any = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>\0".as_ptr() as *const i8),
        );
        let uniform = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>\0".as_ptr() as *const i8
            ),
        );
        let perAxis = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>\0".as_ptr() as *const i8,
            ),
        );
        let calibrated = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.calibrated<f32<-0.998:1.2321>>\0".as_ptr() as *const i8
            ),
        );

        // The parser itself is checked in C++ dialect tests.
        assert!(!mlirTypeIsNull(any), "couldn't parse AnyQuantizedType");
        assert!(
            !mlirTypeIsNull(uniform),
            "couldn't parse UniformQuantizedType"
        );
        assert!(
            !mlirTypeIsNull(perAxis),
            "couldn't parse UniformQuantizedPerAxisType"
        );
        assert!(
            !mlirTypeIsNull(calibrated),
            "couldn't parse CalibratedQuantizedType"
        );

        // CHECK: i8 isa QuantizedType: 0
        eprintln!("i8 isa QuantizedType: {}", mlirTypeIsAQuantizedType(i8Ty));
        // CHECK: any isa QuantizedType: 1
        eprintln!("any isa QuantizedType: {}", mlirTypeIsAQuantizedType(any));
        // CHECK: uniform isa QuantizedType: 1
        eprintln!(
            "uniform isa QuantizedType: {}",
            mlirTypeIsAQuantizedType(uniform)
        );
        // CHECK: perAxis isa QuantizedType: 1
        eprintln!(
            "perAxis isa QuantizedType: {}",
            mlirTypeIsAQuantizedType(perAxis)
        );
        // CHECK: calibrated isa QuantizedType: 1
        eprintln!(
            "calibrated isa QuantizedType: {}",
            mlirTypeIsAQuantizedType(calibrated)
        );

        // CHECK: any isa AnyQuantizedType: 1
        eprintln!(
            "any isa AnyQuantizedType: {}",
            mlirTypeIsAAnyQuantizedType(any)
        );
        // CHECK: uniform isa UniformQuantizedType: 1
        eprintln!(
            "uniform isa UniformQuantizedType: {}",
            mlirTypeIsAUniformQuantizedType(uniform)
        );
        // CHECK: perAxis isa UniformQuantizedPerAxisType: 1
        eprintln!(
            "perAxis isa UniformQuantizedPerAxisType: {}",
            mlirTypeIsAUniformQuantizedPerAxisType(perAxis)
        );
        // CHECK: calibrated isa CalibratedQuantizedType: 1
        eprintln!(
            "calibrated isa CalibratedQuantizedType: {}",
            mlirTypeIsACalibratedQuantizedType(calibrated)
        );

        // CHECK: perAxis isa UniformQuantizedType: 0
        eprintln!(
            "perAxis isa UniformQuantizedType: {}",
            mlirTypeIsAUniformQuantizedType(perAxis)
        );
        // CHECK: uniform isa CalibratedQuantizedType: 0
        eprintln!(
            "uniform isa CalibratedQuantizedType: {}",
            mlirTypeIsACalibratedQuantizedType(uniform)
        );
        eprintln!();
    }
}

// CHECK-LABEL: testAnyQuantizedType
fn testAnyQuantizedType(ctx: MlirContext) {
    unsafe {
        eprintln!("testAnyQuantizedType");

        let anyParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>\0".as_ptr() as *const i8),
        );

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let f32Ty = mlirF32TypeGet(ctx);
        let any = mlirAnyQuantizedTypeGet(mlirQuantizedTypeGetSignedFlag(), i8Ty, f32Ty, -8, 7);

        // CHECK: flags: 1
        eprintln!("flags: {}", mlirQuantizedTypeGetFlags(any));
        // CHECK: signed: 1
        eprintln!("signed: {}", mlirQuantizedTypeIsSigned(any));
        // CHECK: storage type: i8
        eprint!("storage type: ");
        mlirTypeDump(mlirQuantizedTypeGetStorageType(any));
        eprintln!();
        // CHECK: expressed type: f32
        eprint!("expressed type: ");
        mlirTypeDump(mlirQuantizedTypeGetExpressedType(any));
        eprintln!();
        // CHECK: storage min: -8
        eprintln!("storage min: {}", mlirQuantizedTypeGetStorageTypeMin(any));
        // CHECK: storage max: 7
        eprintln!("storage max: {}", mlirQuantizedTypeGetStorageTypeMax(any));
        // CHECK: storage width: 8
        eprintln!(
            "storage width: {}",
            mlirQuantizedTypeGetStorageTypeIntegralWidth(any)
        );
        // CHECK: quantized element type: !quant.any<i8<-8:7>:f32>
        eprint!("quantized element type: ");
        mlirTypeDump(mlirQuantizedTypeGetQuantizedElementType(any));
        eprintln!();

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(anyParsed, any));
        // CHECK: !quant.any<i8<-8:7>:f32>
        mlirTypeDump(any);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testUniformType
fn testUniformType(ctx: MlirContext) {
    unsafe {
        eprintln!("testUniformType");

        let uniformParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.uniform<i8<-8:7>:f32, 0.99872:127>\0".as_ptr() as *const i8
            ),
        );

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let f32Ty = mlirF32TypeGet(ctx);
        let uniform = mlirUniformQuantizedTypeGet(
            mlirQuantizedTypeGetSignedFlag(),
            i8Ty,
            f32Ty,
            0.99872,
            127,
            -8,
            7,
        );

        // CHECK: scale: 0.998720
        eprintln!("scale: {:.6}", mlirUniformQuantizedTypeGetScale(uniform));
        // CHECK: zero point: 127
        eprintln!(
            "zero point: {}",
            mlirUniformQuantizedTypeGetZeroPoint(uniform)
        );
        // CHECK: fixed point: 0
        eprintln!(
            "fixed point: {}",
            mlirUniformQuantizedTypeIsFixedPoint(uniform)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(uniform, uniformParsed));
        // CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
        mlirTypeDump(uniform);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testUniformPerAxisType
fn testUniformPerAxisType(ctx: MlirContext) {
    unsafe {
        eprintln!("testUniformPerAxisType");

        let perAxisParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.uniform<i8:f32:1, {2.0e+2,0.99872:120}>\0".as_ptr() as *const i8,
            ),
        );

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let f32Ty = mlirF32TypeGet(ctx);
        let mut scales = [200.0f64, 0.99872f64];
        let mut zeroPoints = [0i64, 120i64];
        let perAxis = mlirUniformQuantizedPerAxisTypeGet(
            mlirQuantizedTypeGetSignedFlag(),
            i8Ty,
            f32Ty,
            /*nDims=*/ 2,
            scales.as_mut_ptr(),
            zeroPoints.as_mut_ptr(),
            /*quantizedDimension=*/ 1,
            mlirQuantizedTypeGetDefaultMinimumForInteger(
                /*isSigned=*/ 1, /*integralWidth=*/ 8,
            ),
            mlirQuantizedTypeGetDefaultMaximumForInteger(
                /*isSigned=*/ 1, /*integralWidth=*/ 8,
            ),
        );

        // CHECK: num dims: 2
        eprintln!(
            "num dims: {}",
            mlirUniformQuantizedPerAxisTypeGetNumDims(perAxis)
        );
        // CHECK: scale 0: 200.000000
        eprintln!(
            "scale 0: {:.6}",
            mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 0)
        );
        // CHECK: scale 1: 0.998720
        eprintln!(
            "scale 1: {:.6}",
            mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 1)
        );
        // CHECK: zero point 0: 0
        eprintln!(
            "zero point 0: {}",
            mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 0)
        );
        // CHECK: zero point 1: 120
        eprintln!(
            "zero point 1: {}",
            mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 1)
        );
        // CHECK: quantized dim: 1
        eprintln!(
            "quantized dim: {}",
            mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(perAxis)
        );
        // CHECK: fixed point: 0
        eprintln!(
            "fixed point: {}",
            mlirUniformQuantizedPerAxisTypeIsFixedPoint(perAxis)
        );

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(perAxis, perAxisParsed));
        // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
        mlirTypeDump(perAxis);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testCalibratedType
fn testCalibratedType(ctx: MlirContext) {
    unsafe {
        eprintln!("testCalibratedType");

        let calibratedParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.calibrated<f32<-0.998:1.2321>>\0".as_ptr() as *const i8
            ),
        );

        let f32Ty = mlirF32TypeGet(ctx);
        let calibrated = mlirCalibratedQuantizedTypeGet(f32Ty, -0.998, 1.2321);

        // CHECK: min: -0.998000
        eprintln!("min: {:.6}", mlirCalibratedQuantizedTypeGetMin(calibrated));
        // CHECK: max: 1.232100
        eprintln!("max: {:.6}", mlirCalibratedQuantizedTypeGetMax(calibrated));

        // CHECK: equal: 1
        eprintln!("equal: {}", mlirTypeEqual(calibrated, calibratedParsed));
        // CHECK: !quant.calibrated<f32<-0.998:1.232100e+00>>
        mlirTypeDump(calibrated);
        eprint!("\n\n");
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__quant__(), ctx);
        testTypeHierarchy(ctx);
        testAnyQuantizedType(ctx);
        testUniformType(ctx);
        testUniformPerAxisType(ctx);
        testCalibratedType(ctx);
        mlirContextDestroy(ctx);
    }
}
