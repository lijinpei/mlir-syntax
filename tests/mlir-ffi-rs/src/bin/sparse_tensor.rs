// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::AffineMap::*;
use mlir_capi::Dialect_::SparseTensor::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

// CHECK-LABEL: testRoundtripEncoding()
fn testRoundtripEncoding(ctx: MlirContext) -> i32 {
    unsafe {
        eprintln!("testRoundtripEncoding()");
        // clang-format off
        let originalAsm = "#sparse_tensor.encoding<{
map = [s0](d0, d1) -> (s0 : dense, d0 : compressed, d1 : compressed),
posWidth = 32, crdWidth = 64, explicitVal = 1 : i64}>\0"
            .as_ptr() as *const i8;
        // clang-format on
        let originalAttr = mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(originalAsm));
        // CHECK: isa: 1
        eprintln!(
            "isa: {}",
            mlirAttributeIsASparseTensorEncodingAttr(originalAttr)
        );
        let dimToLvl = mlirSparseTensorEncodingAttrGetDimToLvl(originalAttr);
        // CHECK: (d0, d1)[s0] -> (s0, d0, d1)
        mlirAffineMapDump(dimToLvl);
        // CHECK: level_type: 65536
        // CHECK: level_type: 262144
        // CHECK: level_type: 262144
        let lvlToDim = mlirSparseTensorEncodingAttrGetLvlToDim(originalAttr);
        let lvlRank = mlirSparseTensorEncodingGetLvlRank(originalAttr);
        let mut lvlTypes = Vec::new();
        for l in 0..lvlRank {
            lvlTypes.push(mlirSparseTensorEncodingAttrGetLvlType(originalAttr, l));
            eprintln!("level_type: {}", lvlTypes[l as usize]);
        }
        // CHECK: posWidth: 32
        let posWidth = mlirSparseTensorEncodingAttrGetPosWidth(originalAttr);
        eprintln!("posWidth: {}", posWidth);
        // CHECK: crdWidth: 64
        let crdWidth = mlirSparseTensorEncodingAttrGetCrdWidth(originalAttr);
        eprintln!("crdWidth: {}", crdWidth);

        // CHECK: explicitVal: 1 : i64
        let explicitVal = mlirSparseTensorEncodingAttrGetExplicitVal(originalAttr);
        eprint!("explicitVal: ");
        mlirAttributeDump(explicitVal);
        // CHECK: implicitVal: <<NULL ATTRIBUTE>>
        let implicitVal = mlirSparseTensorEncodingAttrGetImplicitVal(originalAttr);
        eprint!("implicitVal: ");
        mlirAttributeDump(implicitVal);

        let newAttr = mlirSparseTensorEncodingAttrGet(
            ctx,
            lvlRank,
            lvlTypes.as_ptr(),
            dimToLvl,
            lvlToDim,
            posWidth,
            crdWidth,
            explicitVal,
            implicitVal,
        );
        mlirAttributeDump(newAttr); // For debugging filecheck output.
                                    // CHECK: equal: 1
        eprintln!("equal: {}", mlirAttributeEqual(originalAttr, newAttr));
        0
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__sparse_tensor__(), ctx);
        if 0 != testRoundtripEncoding(ctx) {
            std::process::exit(1);
        }

        mlirContextDestroy(ctx);
    }
}
