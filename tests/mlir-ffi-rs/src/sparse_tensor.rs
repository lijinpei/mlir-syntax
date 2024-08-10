#![allow(non_snake_case)]

use mlir;
use mlir::AffineMap::*;
use mlir::Dialect_::SparseTensor::*;
use mlir::Support::*;
use mlir::IR::*;

// CHECK-LABEL: testRoundtripEncoding()
fn testRoundtripEncoding(ctx: MlirContext) -> i32 {
    unsafe {
        eprint!("testRoundtripEncoding()\n");
        // clang-format off
        let originalAsm = "#sparse_tensor.encoding<{ 
map = [s0](d0, d1) -> (s0 : dense, d0 : compressed, d1 : compressed), 
posWidth = 32, crdWidth = 64, explicitVal = 1 : i64}>\0"
            .as_ptr() as *const i8;
        // clang-format on
        let originalAttr = mlirAttributeParseGet(ctx, mlirStringRefCreateFromCString(originalAsm));
        // CHECK: isa: 1
        eprint!(
            "isa: {}\n",
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
            eprint!("level_type: {}\n", lvlTypes[l as usize]);
        }
        // CHECK: posWidth: 32
        let posWidth = mlirSparseTensorEncodingAttrGetPosWidth(originalAttr);
        eprint!("posWidth: {}\n", posWidth);
        // CHECK: crdWidth: 64
        let crdWidth = mlirSparseTensorEncodingAttrGetCrdWidth(originalAttr);
        eprint!("crdWidth: {}\n", crdWidth);

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
        eprint!("equal: {}\n", mlirAttributeEqual(originalAttr, newAttr));
        return 0;
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
