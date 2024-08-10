#![allow(non_snake_case)]

use libc;

use mlir;
use mlir::BuiltinAttributes::*;
use mlir::BuiltinTypes::*;
use mlir::Dialect_::LLVM::*;
use mlir::Support::*;
use mlir::IR::*;

use llvm::DebugInfo::*;
use mlir_ffi_rs::common::mlirLogicalResultIsSuccess;

// CHECK-LABEL: testTypeCreation()
fn testTypeCreation(ctx: MlirContext) {
    unsafe {
        eprint!("testTypeCreation()\n");
        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let i32Ty = mlirIntegerTypeGet(ctx, 32);
        let i64Ty = mlirIntegerTypeGet(ctx, 64);

        let ptr_text = "!llvm.ptr\0".as_ptr() as *const i8;
        let ptr = mlirLLVMPointerTypeGet(ctx, 0);
        let ptr_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_text));
        // CHECK: !llvm.ptr: 1
        {
            let ptr_u8 = ptr_text as *const u8;
            let len = libc::strlen(ptr_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!("{}: {}\n", tmp_str, mlirTypeEqual(ptr, ptr_ref));
        }

        let ptr_addr_text = "!llvm.ptr<42>\0".as_ptr() as *const i8;
        let ptr_addr = mlirLLVMPointerTypeGet(ctx, 42);
        let ptr_addr_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(ptr_addr_text));
        // CHECK: !llvm.ptr<42>: 1
        {
            let ptr_u8 = ptr_addr_text as *const u8;
            let len = libc::strlen(ptr_addr_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!("{}: {}\n", tmp_str, mlirTypeEqual(ptr_addr, ptr_addr_ref));
        }

        let voidt_text = "!llvm.void\0".as_ptr() as *const i8;
        let voidt = mlirLLVMVoidTypeGet(ctx);
        let voidt_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(voidt_text));
        // CHECK: !llvm.void: 1
        {
            let ptr_u8 = voidt_text as *const u8;
            let len = libc::strlen(voidt_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!("{}: {}\n", tmp_str, mlirTypeEqual(voidt, voidt_ref));
        }

        let i32_4_text = "!llvm.array<4 x i32>\0".as_ptr() as *const i8;
        let i32_4 = mlirLLVMArrayTypeGet(i32Ty, 4);
        let i32_4_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_4_text));
        // CHECK: !llvm.array<4 x i32>: 1
        {
            let ptr_u8 = i32_4_text as *const u8;
            let len = libc::strlen(i32_4_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!("{}: {}\n", tmp_str, mlirTypeEqual(i32_4, i32_4_ref));
        }

        let i8_i32_i64_text = "!llvm.func<i8 (i32, i64)>\0".as_ptr() as *const i8;
        let i32_i64_arr = [i32Ty, i64Ty];
        let i8_i32_i64 = mlirLLVMFunctionTypeGet(i8Ty, 2, i32_i64_arr.as_ptr(), 0);
        let i8_i32_i64_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i8_i32_i64_text));
        // CHECK: !llvm.func<i8 (i32, i64)>: 1
        {
            let ptr_u8 = i8_i32_i64_text as *const u8;
            let len = libc::strlen(i8_i32_i64_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!(
                "{}: {}\n",
                tmp_str,
                mlirTypeEqual(i8_i32_i64, i8_i32_i64_ref)
            );
        }

        let i32_i64_s_text = "!llvm.struct<(i32, i64)>\0".as_ptr() as *const i8;
        let i32_i64_s = mlirLLVMStructTypeLiteralGet(ctx, 2, i32_i64_arr.as_ptr(), 0);
        let i32_i64_s_ref = mlirTypeParseGet(ctx, mlirStringRefCreateFromCString(i32_i64_s_text));
        // CHECK: !llvm.struct<(i32, i64)>: 1
        {
            let ptr_u8 = i32_i64_s_text as *const u8;
            let len = libc::strlen(i32_i64_s_text);
            let tmp_slice = std::slice::from_raw_parts(ptr_u8, len);
            let tmp_str = std::str::from_utf8_unchecked(tmp_slice);
            eprint!("{}: {}\n", tmp_str, mlirTypeEqual(i32_i64_s, i32_i64_s_ref));
        }
    }
}

// CHECK-LABEL: testStructTypeCreation
fn testStructTypeCreation(ctx: MlirContext) -> i32 {
    unsafe {
        eprint!("testStructTypeCreation\n");

        // CHECK: !llvm.struct<()>
        mlirTypeDump(mlirLLVMStructTypeLiteralGet(
            ctx,
            /*nFieldTypes=*/ 0,
            /*fieldTypes=*/ std::ptr::null_mut(),
            /*isPacked=*/ 0,
        ));

        let i8Ty = mlirIntegerTypeGet(ctx, 8);
        let i32Ty = mlirIntegerTypeGet(ctx, 32);
        let i64Ty = mlirIntegerTypeGet(ctx, 64);
        let i8_i32_i64 = [i8Ty, i32Ty, i64Ty];
        // CHECK: !llvm.struct<(i8, i32, i64)>
        mlirTypeDump(mlirLLVMStructTypeLiteralGet(
            ctx,
            i8_i32_i64.len() as _,
            i8_i32_i64.as_ptr(),
            /*isPacked=*/ 0,
        ));
        // CHECK: !llvm.struct<(i32)>
        mlirTypeDump(mlirLLVMStructTypeLiteralGet(
            ctx, 1, &i32Ty, /*isPacked=*/ 0,
        ));
        let i32_i32 = [i32Ty, i32Ty];
        // CHECK: !llvm.struct<packed (i32, i32)>
        mlirTypeDump(mlirLLVMStructTypeLiteralGet(
            ctx,
            i32_i32.len() as _,
            i32_i32.as_ptr(),
            /*isPacked=*/ 1,
        ));

        let literal = mlirLLVMStructTypeLiteralGet(
            ctx,
            i8_i32_i64.len() as _,
            i8_i32_i64.as_ptr(),
            /*isPacked=*/ 0,
        );
        // CHECK: num elements: 3
        // CHECK: i8
        // CHECK: i32
        // CHECK: i64
        eprint!(
            "num elements: {}\n",
            mlirLLVMStructTypeGetNumElementTypes(literal)
        );
        mlirTypeDump(mlirLLVMStructTypeGetElementType(literal, 0));
        mlirTypeDump(mlirLLVMStructTypeGetElementType(literal, 1));
        mlirTypeDump(mlirLLVMStructTypeGetElementType(literal, 2));

        if 0 == mlirTypeEqual(
            mlirLLVMStructTypeLiteralGet(ctx, 1, &i32Ty, /*isPacked=*/ 0),
            mlirLLVMStructTypeLiteralGet(ctx, 1, &i32Ty, /*isPacked=*/ 0),
        ) {
            return 1;
        }
        if 0 != mlirTypeEqual(
            mlirLLVMStructTypeLiteralGet(ctx, 1, &i32Ty, /*isPacked=*/ 0),
            mlirLLVMStructTypeLiteralGet(ctx, 1, &i64Ty, /*isPacked=*/ 0),
        ) {
            return 2;
        }

        // CHECK: !llvm.struct<"foo", opaque>
        // CHECK: !llvm.struct<"bar", opaque>
        mlirTypeDump(mlirLLVMStructTypeIdentifiedGet(
            ctx,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        ));
        mlirTypeDump(mlirLLVMStructTypeIdentifiedGet(
            ctx,
            mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
        ));

        if 0 == mlirTypeEqual(
            mlirLLVMStructTypeIdentifiedGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            ),
            mlirLLVMStructTypeIdentifiedGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            ),
        ) {
            return 3;
        }
        if 0 != mlirTypeEqual(
            mlirLLVMStructTypeIdentifiedGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            ),
            mlirLLVMStructTypeIdentifiedGet(
                ctx,
                mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
            ),
        ) {
            return 4;
        }

        let fooStruct = mlirLLVMStructTypeIdentifiedGet(
            ctx,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        );
        let name = mlirLLVMStructTypeGetIdentifier(fooStruct);
        if 0 != libc::memcmp(name.data as _, "foo\0".as_ptr() as _, name.length as _) {
            return 5;
        }
        if 0 == mlirLLVMStructTypeIsOpaque(fooStruct) {
            return 6;
        }

        let i32_i64 = [i32Ty, i64Ty];
        let mut result = mlirLLVMStructTypeSetBody(
            fooStruct,
            i32_i64.len() as _,
            i32_i64.as_ptr(),
            /*isPacked=*/ 0,
        );
        if !mlirLogicalResultIsSuccess(result) {
            return 7;
        }

        // CHECK: !llvm.struct<"foo", (i32, i64)>
        mlirTypeDump(fooStruct);
        if 0 != mlirLLVMStructTypeIsOpaque(fooStruct) {
            return 8;
        }
        if 0 != mlirLLVMStructTypeIsPacked(fooStruct) {
            return 9;
        }
        if 0 == mlirTypeEqual(
            mlirLLVMStructTypeIdentifiedGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            ),
            fooStruct,
        ) {
            return 10;
        }

        let barStruct = mlirLLVMStructTypeIdentifiedGet(
            ctx,
            mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
        );
        result = mlirLLVMStructTypeSetBody(barStruct, 1, &i32Ty, /*isPacked=*/ 1);
        if !mlirLogicalResultIsSuccess(result) {
            return 11;
        }

        // CHECK: !llvm.struct<"bar", packed (i32)>
        mlirTypeDump(barStruct);
        if 0 == mlirLLVMStructTypeIsPacked(barStruct) {
            return 12;
        }

        // Same body, should succeed.
        result = mlirLLVMStructTypeSetBody(
            fooStruct,
            i32_i64.len() as _,
            i32_i64.as_ptr(),
            /*isPacked=*/ 0,
        );
        if !mlirLogicalResultIsSuccess(result) {
            return 13;
        }

        // Different body, should fail.
        result = mlirLLVMStructTypeSetBody(fooStruct, 1, &i32Ty, /*isPacked=*/ 0);
        if mlirLogicalResultIsSuccess(result) {
            return 14;
        }

        // Packed flag differs, should fail.
        result = mlirLLVMStructTypeSetBody(barStruct, 1, &i32Ty, /*isPacked=*/ 0);
        if mlirLogicalResultIsSuccess(result) {
            return 15;
        }

        // Should have a different name.
        // CHECK: !llvm.struct<"foo{{[^"]+}}
        mlirTypeDump(mlirLLVMStructTypeIdentifiedNewGet(
            ctx,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            /*nFieldTypes=*/ 0,
            /*fieldTypes=*/ std::ptr::null_mut(),
            /*isPacked=*/ 0,
        ));

        // Two freshly created "new" types must differ.
        if 0 != mlirTypeEqual(
            mlirLLVMStructTypeIdentifiedNewGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
                /*nFieldTypes=*/ 0,
                /*fieldTypes=*/ std::ptr::null_mut(),
                /*isPacked=*/ 0,
            ),
            mlirLLVMStructTypeIdentifiedNewGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
                /*nFieldTypes=*/ 0,
                /*fieldTypes=*/ std::ptr::null_mut(),
                /*isPacked=*/ 0,
            ),
        ) {
            return 16;
        }

        let opaque = mlirLLVMStructTypeOpaqueGet(
            ctx,
            mlirStringRefCreateFromCString("opaque\0".as_ptr() as *const i8),
        );
        // CHECK: !llvm.struct<"opaque", opaque>
        mlirTypeDump(opaque);
        if 0 == mlirLLVMStructTypeIsOpaque(opaque) {
            return 17;
        }

        return 0;
    }
}

// CHECK-LABEL: testLLVMAttributes
fn testLLVMAttributes(ctx: MlirContext) {
    unsafe {
        eprint!("testLLVMAttributes\n");

        // CHECK: #llvm.linkage<internal>
        mlirAttributeDump(mlirLLVMLinkageAttrGet(ctx, MlirLLVMLinkageInternal));
        // CHECK: #llvm.cconv<ccc>
        mlirAttributeDump(mlirLLVMCConvAttrGet(ctx, MlirLLVMCConvC));
        // CHECK: #llvm<comdat any>
        mlirAttributeDump(mlirLLVMComdatAttrGet(ctx, MlirLLVMComdatAny));
    }
}

// CHECK-LABEL: testDebugInfoAttributes
fn testDebugInfoAttributes(ctx: MlirContext) {
    unsafe {
        eprint!("testDebugInfoAttributes\n");

        let foo = mlirStringAttrGet(
            ctx,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        );
        let bar = mlirStringAttrGet(
            ctx,
            mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
        );

        let none = mlirUnitAttrGet(ctx);
        let id = mlirDisctinctAttrCreate(none);
        let recId0 = mlirDisctinctAttrCreate(none);
        let recId1 = mlirDisctinctAttrCreate(none);

        // CHECK: #llvm.di_null_type
        mlirAttributeDump(mlirLLVMDINullTypeAttrGet(ctx));

        // CHECK: #llvm.di_basic_type<name = "foo", sizeInBits =
        // CHECK-SAME: 64, encoding = DW_ATE_signed>
        let di_type = mlirLLVMDIBasicTypeAttrGet(ctx, 0, foo, 64, MlirLLVMTypeEncodingSigned);
        mlirAttributeDump(di_type);

        let file = mlirLLVMDIFileAttrGet(ctx, foo, bar);

        // CHECK: #llvm.di_file<"foo" in "bar">
        mlirAttributeDump(file);

        let compile_unit = mlirLLVMDICompileUnitAttrGet(
            ctx,
            id,
            LLVMDWARFSourceLanguageC99,
            file,
            foo,
            0,
            MlirLLVMDIEmissionKindFull,
            MlirLLVMDINameTableKindDefault,
        );

        // CHECK: #llvm.di_compile_unit<{{.*}}>
        mlirAttributeDump(compile_unit);

        let di_module = mlirLLVMDIModuleAttrGet(
            ctx,
            file,
            compile_unit,
            foo,
            mlirStringAttrGet(
                ctx,
                mlirStringRefCreateFromCString("\0".as_ptr() as *const i8),
            ),
            bar,
            foo,
            1,
            0,
        );
        // CHECK: #llvm.di_module<{{.*}}>
        mlirAttributeDump(di_module);

        // CHECK: #llvm.di_compile_unit<{{.*}}>
        mlirAttributeDump(mlirLLVMDIModuleAttrGetScope(di_module));

        // CHECK: 1 : i32
        mlirAttributeDump(mlirLLVMDIFlagsAttrGet(ctx, 0x1));

        // CHECK: #llvm.di_lexical_block<{{.*}}>
        mlirAttributeDump(mlirLLVMDILexicalBlockAttrGet(ctx, compile_unit, file, 1, 2));

        // CHECK: #llvm.di_lexical_block_file<{{.*}}>
        mlirAttributeDump(mlirLLVMDILexicalBlockFileAttrGet(
            ctx,
            compile_unit,
            file,
            3,
        ));

        // CHECK: #llvm.di_local_variable<{{.*}}>
        let local_var =
            mlirLLVMDILocalVariableAttrGet(ctx, compile_unit, foo, file, 1, 0, 8, di_type, 0);
        mlirAttributeDump(local_var);
        // CHECK: #llvm.di_derived_type<{{.*}}>
        // CHECK-NOT: dwarfAddressSpace

        // FIXME: MLIR_CAPI_DWARF_ADDRESS_SPACE_NULL
        mlirAttributeDump(mlirLLVMDIDerivedTypeAttrGet(
            ctx, 0, bar, di_type, 64, 8, 0, -1, di_type,
        ));

        // CHECK: #llvm.di_derived_type<{{.*}} dwarfAddressSpace = 3{{.*}}>
        mlirAttributeDump(mlirLLVMDIDerivedTypeAttrGet(
            ctx, 0, bar, di_type, 64, 8, 0, 3, di_type,
        ));

        let subroutine_type = mlirLLVMDISubroutineTypeAttrGet(ctx, 0x0, 1, &di_type);

        // CHECK: #llvm.di_subroutine_type<{{.*}}>
        mlirAttributeDump(subroutine_type);

        let di_subprogram_self_rec = mlirLLVMDISubprogramAttrGetRecSelf(recId0);
        let di_imported_entity = mlirLLVMDIImportedEntityAttrGet(
            ctx,
            0,
            di_subprogram_self_rec,
            di_module,
            file,
            1,
            foo,
            1,
            &local_var,
        );

        mlirAttributeDump(di_imported_entity);
        // CHECK: #llvm.di_imported_entity<{{.*}}>

        let di_annotation = mlirLLVMDIAnnotationAttrGet(
            ctx,
            mlirStringAttrGet(
                ctx,
                mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
            ),
            mlirStringAttrGet(
                ctx,
                mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
            ),
        );

        mlirAttributeDump(di_annotation);
        // CHECK: #llvm.di_annotation<{{.*}}>

        let di_subprogram = mlirLLVMDISubprogramAttrGet(
            ctx,
            recId0,
            0,
            id,
            compile_unit,
            compile_unit,
            foo,
            bar,
            file,
            1,
            2,
            0,
            subroutine_type,
            1,
            &di_imported_entity,
            1,
            &di_annotation,
        );
        // CHECK: #llvm.di_subprogram<{{.*}}>
        mlirAttributeDump(di_subprogram);

        // CHECK: #llvm.di_compile_unit<{{.*}}>
        mlirAttributeDump(mlirLLVMDISubprogramAttrGetScope(di_subprogram));

        // CHECK: #llvm.di_file<{{.*}}>
        mlirAttributeDump(mlirLLVMDISubprogramAttrGetFile(di_subprogram));

        // CHECK: #llvm.di_subroutine_type<{{.*}}>
        mlirAttributeDump(mlirLLVMDISubprogramAttrGetType(di_subprogram));

        let tmp_data: u64 = 1;
        let expression_elem = mlirLLVMDIExpressionElemAttrGet(ctx, 1, 1, &tmp_data);

        // CHECK: #llvm<di_expression_elem(1)>
        mlirAttributeDump(expression_elem);

        let expression = mlirLLVMDIExpressionAttrGet(ctx, 1, &expression_elem);
        // CHECK: #llvm.di_expression<[(1)]>
        mlirAttributeDump(expression);

        let string_type = mlirLLVMDIStringTypeAttrGet(
            ctx,
            0x0,
            foo,
            16,
            0,
            local_var,
            expression,
            expression,
            MlirLLVMTypeEncodingSigned,
        );
        // CHECK: #llvm.di_string_type<{{.*}}>
        mlirAttributeDump(string_type);

        // CHECK: #llvm.di_composite_type<recId = {{.*}}, isRecSelf = true>
        mlirAttributeDump(mlirLLVMDICompositeTypeAttrGetRecSelf(recId1));

        // CHECK: #llvm.di_composite_type<{{.*}}>
        mlirAttributeDump(mlirLLVMDICompositeTypeAttrGet(
            ctx,
            recId1,
            0,
            0,
            foo,
            file,
            1,
            compile_unit,
            di_type,
            0,
            64,
            8,
            1,
            &di_type,
            expression,
            expression,
            expression,
            expression,
        ));
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirDialectHandleRegisterDialect(mlirGetDialectHandle__llvm__(), ctx);
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("llvm\0".as_ptr() as *const i8),
        );
        testTypeCreation(ctx);
        let result = testStructTypeCreation(ctx);
        testLLVMAttributes(ctx);
        testDebugInfoAttributes(ctx);
        mlirContextDestroy(ctx);
        if 0 != result {
            eprint!("FAILED: code {}", result);
            std::process::exit(result);
        }
    }
}
