#![allow(non_snake_case)]

use mlir;
use mlir::BuiltinTypes::*;
use mlir::Dialect_::Quant::*;
use mlir::Support::*;
use mlir::IR::*;

use mlir_ffi_rs::common::mlirTypeIsNull;

// CHECK-LABEL: testTypeHierarchy
fn testTypeHierarchy(ctx: MlirContext) {
    unsafe {
        eprint!("testTypeHierarchy\n");

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
        eprint!("i8 isa QuantizedType: {}\n", mlirTypeIsAQuantizedType(i8Ty));
        // CHECK: any isa QuantizedType: 1
        eprint!("any isa QuantizedType: {}\n", mlirTypeIsAQuantizedType(any));
        // CHECK: uniform isa QuantizedType: 1
        eprint!(
            "uniform isa QuantizedType: {}\n",
            mlirTypeIsAQuantizedType(uniform)
        );
        // CHECK: perAxis isa QuantizedType: 1
        eprint!(
            "perAxis isa QuantizedType: {}\n",
            mlirTypeIsAQuantizedType(perAxis)
        );
        // CHECK: calibrated isa QuantizedType: 1
        eprint!(
            "calibrated isa QuantizedType: {}\n",
            mlirTypeIsAQuantizedType(calibrated)
        );

        // CHECK: any isa AnyQuantizedType: 1
        eprint!(
            "any isa AnyQuantizedType: {}\n",
            mlirTypeIsAAnyQuantizedType(any)
        );
        // CHECK: uniform isa UniformQuantizedType: 1
        eprint!(
            "uniform isa UniformQuantizedType: {}\n",
            mlirTypeIsAUniformQuantizedType(uniform)
        );
        // CHECK: perAxis isa UniformQuantizedPerAxisType: 1
        eprint!(
            "perAxis isa UniformQuantizedPerAxisType: {}\n",
            mlirTypeIsAUniformQuantizedPerAxisType(perAxis)
        );
        // CHECK: calibrated isa CalibratedQuantizedType: 1
        eprint!(
            "calibrated isa CalibratedQuantizedType: {}\n",
            mlirTypeIsACalibratedQuantizedType(calibrated)
        );

        // CHECK: perAxis isa UniformQuantizedType: 0
        eprint!(
            "perAxis isa UniformQuantizedType: {}\n",
            mlirTypeIsAUniformQuantizedType(perAxis)
        );
        // CHECK: uniform isa CalibratedQuantizedType: 0
        eprint!(
            "uniform isa CalibratedQuantizedType: {}\n",
            mlirTypeIsACalibratedQuantizedType(uniform)
        );
        eprint!("\n");
    }
}

// CHECK-LABEL: testAnyQuantizedType
fn testAnyQuantizedType(ctx: MlirContext) {
    unsafe {
        eprint!("testAnyQuantizedType\n");

        let anyParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("!quant.any<i8<-8:7>:f32>\0".as_ptr() as *const i8),
        );

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let f32Ty = mlirF32TypeGet(ctx);
        let any = mlirAnyQuantizedTypeGet(mlirQuantizedTypeGetSignedFlag(), i8Ty, f32Ty, -8, 7);

        // CHECK: flags: 1
        eprint!("flags: {}\n", mlirQuantizedTypeGetFlags(any));
        // CHECK: signed: 1
        eprint!("signed: {}\n", mlirQuantizedTypeIsSigned(any));
        // CHECK: storage type: i8
        eprint!("storage type: ");
        mlirTypeDump(mlirQuantizedTypeGetStorageType(any));
        eprint!("\n");
        // CHECK: expressed type: f32
        eprint!("expressed type: ");
        mlirTypeDump(mlirQuantizedTypeGetExpressedType(any));
        eprint!("\n");
        // CHECK: storage min: -8
        eprint!("storage min: {}\n", mlirQuantizedTypeGetStorageTypeMin(any));
        // CHECK: storage max: 7
        eprint!("storage max: {}\n", mlirQuantizedTypeGetStorageTypeMax(any));
        // CHECK: storage width: 8
        eprint!(
            "storage width: {}\n",
            mlirQuantizedTypeGetStorageTypeIntegralWidth(any)
        );
        // CHECK: quantized element type: !quant.any<i8<-8:7>:f32>
        eprint!("quantized element type: ");
        mlirTypeDump(mlirQuantizedTypeGetQuantizedElementType(any));
        eprint!("\n");

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(anyParsed, any));
        // CHECK: !quant.any<i8<-8:7>:f32>
        mlirTypeDump(any);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testUniformType
fn testUniformType(ctx: MlirContext) {
    unsafe {
        eprint!("testUniformType\n");

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
        eprint!("scale: {:.6}\n", mlirUniformQuantizedTypeGetScale(uniform));
        // CHECK: zero point: 127
        eprint!(
            "zero point: {}\n",
            mlirUniformQuantizedTypeGetZeroPoint(uniform)
        );
        // CHECK: fixed point: 0
        eprint!(
            "fixed point: {}\n",
            mlirUniformQuantizedTypeIsFixedPoint(uniform)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(uniform, uniformParsed));
        // CHECK: !quant.uniform<i8<-8:7>:f32, 9.987200e-01:127>
        mlirTypeDump(uniform);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testUniformPerAxisType
fn testUniformPerAxisType(ctx: MlirContext) {
    unsafe {
        eprint!("testUniformPerAxisType\n");

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
        eprint!(
            "num dims: {}\n",
            mlirUniformQuantizedPerAxisTypeGetNumDims(perAxis)
        );
        // CHECK: scale 0: 200.000000
        eprint!(
            "scale 0: {:.6}\n",
            mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 0)
        );
        // CHECK: scale 1: 0.998720
        eprint!(
            "scale 1: {:.6}\n",
            mlirUniformQuantizedPerAxisTypeGetScale(perAxis, 1)
        );
        // CHECK: zero point 0: 0
        eprint!(
            "zero point 0: {}\n",
            mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 0)
        );
        // CHECK: zero point 1: 120
        eprint!(
            "zero point 1: {}\n",
            mlirUniformQuantizedPerAxisTypeGetZeroPoint(perAxis, 1)
        );
        // CHECK: quantized dim: 1
        eprint!(
            "quantized dim: {}\n",
            mlirUniformQuantizedPerAxisTypeGetQuantizedDimension(perAxis)
        );
        // CHECK: fixed point: 0
        eprint!(
            "fixed point: {}\n",
            mlirUniformQuantizedPerAxisTypeIsFixedPoint(perAxis)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(perAxis, perAxisParsed));
        // CHECK: !quant.uniform<i8:f32:1, {2.000000e+02,9.987200e-01:120}>
        mlirTypeDump(perAxis);
        eprint!("\n\n");
    }
}

// CHECK-LABEL: testCalibratedType
fn testCalibratedType(ctx: MlirContext) {
    unsafe {
        eprint!("testCalibratedType\n");

        let calibratedParsed = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "!quant.calibrated<f32<-0.998:1.2321>>\0".as_ptr() as *const i8
            ),
        );

        let f32Ty = mlirF32TypeGet(ctx);
        let calibrated = mlirCalibratedQuantizedTypeGet(f32Ty, -0.998, 1.2321);

        // CHECK: min: -0.998000
        eprint!(
            "min: {:.6}\n",
            mlirCalibratedQuantizedTypeGetMin(calibrated)
        );
        // CHECK: max: 1.232100
        eprint!(
            "max: {:.6}\n",
            mlirCalibratedQuantizedTypeGetMax(calibrated)
        );

        // CHECK: equal: 1
        eprint!("equal: {}\n", mlirTypeEqual(calibrated, calibratedParsed));
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
