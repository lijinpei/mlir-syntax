// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::AffineExpr::*;
use mlir_capi::AffineMap::*;
use mlir_capi::BuiltinAttributes::*;
use mlir_capi::BuiltinTypes::*;
use mlir_capi::Diagnostics::*;
use mlir_capi::Dialect_::Func::*;
use mlir_capi::IntegerSet::*;
use mlir_capi::RegisterEverything::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

struct ModuleStats {
    pub numOperations: i64,
    pub numAttributes: i64,
    pub numBlocks: i64,
    pub numRegions: i64,
    pub numValues: i64,
    pub numBlockArguments: i64,
    pub numOpResults: i64,
}

fn collectStatsSingle(head: &mut Vec<MlirOperation>, stats: &mut ModuleStats) -> i32 {
    unsafe {
        let operation = head.pop().unwrap();
        stats.numOperations += 1;
        stats.numValues += mlirOperationGetNumResults(operation);
        stats.numAttributes += mlirOperationGetNumAttributes(operation);

        let numRegions = mlirOperationGetNumRegions(operation);
        stats.numRegions += numRegions;

        let numResults = mlirOperationGetNumResults(operation);
        for i in 0..numResults {
            let result = mlirOperationGetResult(operation, i);
            if 0 == mlirValueIsAOpResult(result) {
                return 1;
            }
            if 1 == mlirValueIsABlockArgument(result) {
                return 2;
            }
            if 0 == mlirOperationEqual(operation, mlirOpResultGetOwner(result)) {
                return 3;
            }
            if i != mlirOpResultGetResultNumber(result) {
                return 4;
            }
            stats.numOpResults += 1;
        }

        let mut region = mlirOperationGetFirstRegion(operation);
        while region.ptr != std::ptr::null_mut() {
            let mut block = mlirRegionGetFirstBlock(region);
            while block.ptr != std::ptr::null_mut() {
                stats.numBlocks += 1;
                let numArgs = mlirBlockGetNumArguments(block);
                stats.numValues += numArgs;
                let mut j = 0;
                while j < numArgs {
                    let arg = mlirBlockGetArgument(block, j);
                    if 0 == mlirValueIsABlockArgument(arg) {
                        return 5;
                    }
                    if 1 == mlirValueIsAOpResult(arg) {
                        return 6;
                    }
                    if 0 == mlirBlockEqual(block, mlirBlockArgumentGetOwner(arg)) {
                        return 7;
                    }
                    if j != mlirBlockArgumentGetArgNumber(arg) {
                        return 8;
                    }
                    stats.numBlockArguments += 1;
                    j += 1;
                }

                let mut child = mlirBlockGetFirstOperation(block);
                while child.ptr != std::ptr::null_mut() {
                    head.push(child);
                    child = mlirOperationGetNextInBlock(child);
                }
                block = mlirBlockGetNextInRegion(block);
            }
            region = mlirRegionGetNextInOperation(region);
        }
        0
    }
}

fn registerAllUpstreamDialects(ctx: MlirContext) {
    unsafe {
        let registry = mlirDialectRegistryCreate();
        mlirRegisterAllDialects(registry);
        mlirContextAppendDialectRegistry(ctx, registry);
        mlirDialectRegistryDestroy(registry);
    }
}

fn populateLoopBody(
    ctx: MlirContext,
    loopBody: MlirBlock,
    location: MlirLocation,
    funcBody: MlirBlock,
) {
    unsafe {
        let iv = mlirBlockGetArgument(loopBody, 0);
        let funcArg0 = mlirBlockGetArgument(funcBody, 0);
        let funcArg1 = mlirBlockGetArgument(funcBody, 1);

        let f32Type = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("f32\0".as_ptr() as *const i8),
        );
        let mut loadLHSState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("memref.load\0".as_ptr() as *const i8),
            location,
        );
        let loadLHSOperands = [funcArg0, iv];
        mlirOperationStateAddOperands(&mut loadLHSState, 2, loadLHSOperands.as_ptr());
        mlirOperationStateAddResults(&mut loadLHSState, 1, &f32Type);
        let loadLHS = mlirOperationCreate(&mut loadLHSState);
        mlirBlockAppendOwnedOperation(loopBody, loadLHS);
        let mut loadRHSState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("memref.load\0".as_ptr() as *const i8),
            location,
        );
        let loadRHSOperands = [funcArg1, iv];
        mlirOperationStateAddOperands(&mut loadRHSState, 2, loadRHSOperands.as_ptr());
        mlirOperationStateAddResults(&mut loadRHSState, 1, &f32Type);
        let loadRHS = mlirOperationCreate(&mut loadRHSState);
        mlirBlockAppendOwnedOperation(loopBody, loadRHS);

        let mut addState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.addf\0".as_ptr() as *const i8),
            location,
        );
        let addOperands = [
            mlirOperationGetResult(loadLHS, 0),
            mlirOperationGetResult(loadRHS, 0),
        ];
        mlirOperationStateAddOperands(&mut addState, 2, addOperands.as_ptr());
        mlirOperationStateAddResults(&mut addState, 1, &f32Type);
        let add = mlirOperationCreate(&mut addState);
        mlirBlockAppendOwnedOperation(loopBody, add);

        let mut storeState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("memref.store\0".as_ptr() as *const i8),
            location,
        );
        let storeOperands = [mlirOperationGetResult(add, 0), funcArg0, iv];
        mlirOperationStateAddOperands(&mut storeState, 3, storeOperands.as_ptr());
        let store = mlirOperationCreate(&mut storeState);
        mlirBlockAppendOwnedOperation(loopBody, store);

        let mut yieldState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("scf.yield\0".as_ptr() as *const i8),
            location,
        );
        let r#yield = mlirOperationCreate(&mut yieldState);
        mlirBlockAppendOwnedOperation(loopBody, r#yield);
    }
}

fn makeAndDumpAdd(ctx: MlirContext, location: MlirLocation) -> MlirModule {
    unsafe {
        let r#mod = mlirModuleCreateEmpty(location);
        let moduleBody = mlirModuleGetBody(r#mod);
        let memrefType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("memref<?xf32>\0".as_ptr() as *const i8),
        );
        let funcBodyArgTypes = [memrefType, memrefType];
        let funcBodyArgLocs = [location, location];
        let funcBodyRegion = mlirRegionCreate();
        let funcBody = mlirBlockCreate(
            funcBodyArgTypes.len() as i64,
            funcBodyArgTypes.as_ptr(),
            funcBodyArgLocs.as_ptr(),
        );
        mlirRegionAppendOwnedBlock(funcBodyRegion, funcBody);
        let funcTypeAttr = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString(
                "(memref<?xf32>, memref<?xf32>) -> ()\0".as_ptr() as *const i8
            ),
        );
        let funcNameAttr = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("\"add\"\0".as_ptr() as *const i8),
        );
        let funcAttrs = [
            mlirNamedAttributeGet(
                mlirIdentifierGet(
                    ctx,
                    mlirStringRefCreateFromCString("function_type\0".as_ptr() as *const i8),
                ),
                funcTypeAttr,
            ),
            mlirNamedAttributeGet(
                mlirIdentifierGet(
                    ctx,
                    mlirStringRefCreateFromCString("sym_name\0".as_ptr() as *const i8),
                ),
                funcNameAttr,
            ),
        ];
        let mut funcState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("func.func\0".as_ptr() as *const i8),
            location,
        );
        mlirOperationStateAddAttributes(&mut funcState, 2, funcAttrs.as_ptr());
        mlirOperationStateAddOwnedRegions(&mut funcState, 1, &funcBodyRegion);
        let func = mlirOperationCreate(&mut funcState);
        mlirBlockInsertOwnedOperation(moduleBody, 0, func);

        let mut indexType = mlirTypeParseGet(
            ctx,
            mlirStringRefCreateFromCString("index\0".as_ptr() as *const i8),
        );
        let indexZeroLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("0 : index\0".as_ptr() as *const i8),
        );
        let indexZeroValueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            indexZeroLiteral,
        );
        let mut constZeroState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            location,
        );
        mlirOperationStateAddResults(&mut constZeroState, 1, &mut indexType);
        mlirOperationStateAddAttributes(&mut constZeroState, 1, &indexZeroValueAttr);
        let constZero = mlirOperationCreate(&mut constZeroState);
        mlirBlockAppendOwnedOperation(funcBody, constZero);

        let funcArg0 = mlirBlockGetArgument(funcBody, 0);
        let constZeroValue = mlirOperationGetResult(constZero, 0);
        let dimOperands = [funcArg0, constZeroValue];
        let mut dimState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("memref.dim\0".as_ptr() as *const i8),
            location,
        );
        mlirOperationStateAddOperands(&mut dimState, 2, dimOperands.as_ptr());
        mlirOperationStateAddResults(&mut dimState, 1, &indexType);
        let dim = mlirOperationCreate(&mut dimState);
        mlirBlockAppendOwnedOperation(funcBody, dim);

        let loopBodyRegion = mlirRegionCreate();
        let loopBody = mlirBlockCreate(0, std::ptr::null(), std::ptr::null());
        mlirBlockAddArgument(loopBody, indexType, location);
        mlirRegionAppendOwnedBlock(loopBodyRegion, loopBody);

        let indexOneLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("1 : index\0".as_ptr() as *const i8),
        );
        let indexOneValueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            indexOneLiteral,
        );
        let mut constOneState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            location,
        );
        mlirOperationStateAddResults(&mut constOneState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constOneState, 1, &indexOneValueAttr);
        let constOne = mlirOperationCreate(&mut constOneState);
        mlirBlockAppendOwnedOperation(funcBody, constOne);

        let dimValue = mlirOperationGetResult(dim, 0);
        let constOneValue = mlirOperationGetResult(constOne, 0);
        let loopOperands = [constZeroValue, dimValue, constOneValue];
        let mut loopState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("scf.for\0".as_ptr() as *const i8),
            location,
        );
        mlirOperationStateAddOperands(&mut loopState, 3, loopOperands.as_ptr());
        mlirOperationStateAddOwnedRegions(&mut loopState, 1, &loopBodyRegion);
        let r#loop = mlirOperationCreate(&mut loopState);
        mlirBlockAppendOwnedOperation(funcBody, r#loop);

        populateLoopBody(ctx, loopBody, location, funcBody);

        let mut retState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("func.return\0".as_ptr() as *const i8),
            location,
        );
        let ret = mlirOperationCreate(&mut retState);
        mlirBlockAppendOwnedOperation(funcBody, ret);

        let moduleOp = mlirModuleGetOperation(r#mod);
        mlirOperationDump(moduleOp);

        r#mod
    }
}

pub unsafe extern "C" fn printToStderr(str: MlirStringRef, _: *mut u8) -> u8 {
    libc::write(1, str.data as *const libc::c_void, str.length as usize);
    0
}
fn constructAndTraverseIr(ctx: MlirContext) -> i32 {
    unsafe {
        let location = mlirLocationUnknownGet(ctx);
        let r#mod = makeAndDumpAdd(ctx, location);
        let modOp = mlirModuleGetOperation(r#mod);
        assert!(r#mod.ptr != std::ptr::null());
        // CHECK: module {
        // CHECK:   func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
        // CHECK:     %[[C0:.*]] = arith.constant 0 : index
        // CHECK:     %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
        // CHECK:     %[[C1:.*]] = arith.constant 1 : index
        // CHECK:     scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
        // CHECK:       %[[LHS:.*]] = memref.load %[[ARG0]][%[[I]]] : memref<?xf32>
        // CHECK:       %[[RHS:.*]] = memref.load %[[ARG1]][%[[I]]] : memref<?xf32>
        // CHECK:       %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
        // CHECK:       memref.store %[[SUM]], %[[ARG0]][%[[I]]] : memref<?xf32>
        // CHECK:     }
        // CHECK:     return
        // CHECK:   }
        // CHECK: }
        let errcode = collectStats(modOp);
        if 0 != errcode {
            return errcode;
        }
        printFirstOfEach(ctx, modOp);
        mlirModuleDestroy(r#mod);
    }
    0
}

fn collectStats(operation: MlirOperation) -> i32 {
    let mut head = vec![operation];
    let mut stats = ModuleStats {
        numOperations: 0,
        numAttributes: 0,
        numBlocks: 0,
        numRegions: 0,
        numValues: 0,
        numBlockArguments: 0,
        numOpResults: 0,
    };

    while !head.is_empty() {
        let retval = collectStatsSingle(&mut head, &mut stats);
        if 0 != retval {
            return retval;
        }
    }

    if stats.numValues != stats.numBlockArguments + stats.numOpResults {
        return 100;
    }

    eprintln!("@stats");
    eprintln!("Number of operations: {}", stats.numOperations);
    eprintln!("Number of attributes: {}", stats.numAttributes);
    eprintln!("Number of blocks: {}", stats.numBlocks);
    eprintln!("Number of regions: {}", stats.numRegions);
    eprintln!("Number of values: {}", stats.numValues);
    eprintln!("Number of block arguments: {}", stats.numBlockArguments);
    eprintln!("Number of op results: {}", stats.numOpResults);
    // clang-format off
    // CHECK-LABEL: @stats
    // CHECK: Number of operations: 12
    // CHECK: Number of attributes: 6
    // CHECK: Number of blocks: 3
    // CHECK: Number of regions: 3
    // CHECK: Number of values: 9
    // CHECK: Number of block arguments: 3
    // CHECK: Number of op results: 6
    // clang-format on
    0
}

fn printFirstOfEach(ctx: MlirContext, mut operation: MlirOperation) {
    unsafe {
        // Assuming we are given a module, go to the first operation of the first
        // function.
        let mut region = mlirOperationGetRegion(operation, 0);
        let mut block = mlirRegionGetFirstBlock(region);
        let function = mlirBlockGetFirstOperation(block);
        region = mlirOperationGetRegion(function, 0);
        let parentOperation = function;
        block = mlirRegionGetFirstBlock(region);
        operation = mlirBlockGetFirstOperation(block);
        assert!(std::ptr::null() == mlirModuleFromOperation(operation).ptr);

        // Verify that parent operation and block report correctly.
        // CHECK: Parent operation eq: 1
        eprintln!(
            "Parent operation eq: {}",
            mlirOperationEqual(mlirOperationGetParentOperation(operation), parentOperation)
        );
        // CHECK: Block eq: 1
        eprintln!(
            "Block eq: {}",
            mlirBlockEqual(mlirOperationGetBlock(operation), block)
        );
        // CHECK: Block parent operation eq: 1
        eprintln!(
            "Block parent operation eq: {}",
            mlirOperationEqual(mlirBlockGetParentOperation(block), parentOperation)
        );
        // CHECK: Block parent region eq: 1
        eprintln!(
            "Block parent region eq: {}",
            mlirRegionEqual(mlirBlockGetParentRegion(block), region)
        );

        // In the module we created, the first operation of the first function is
        // an "memref.dim", which has an attribute and a single result that we can
        // use to test the printing mechanism.
        mlirBlockPrint(
            block,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        eprint!("First operation: ");
        mlirOperationPrint(
            operation,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // clang-format off
        // CHECK:   %[[C0:.*]] = arith.constant 0 : index
        // CHECK:   %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
        // CHECK:   %[[C1:.*]] = arith.constant 1 : index
        // CHECK:   scf.for %[[I:.*]] = %[[C0]] to %[[DIM]] step %[[C1]] {
        // CHECK:     %[[LHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
        // CHECK:     %[[RHS:.*]] = memref.load %{{.*}}[%[[I]]] : memref<?xf32>
        // CHECK:     %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
        // CHECK:     memref.store %[[SUM]], %{{.*}}[%[[I]]] : memref<?xf32>
        // CHECK:   }
        // CHECK: return
        // CHECK: First operation: {{.*}} = arith.constant 0 : index
        // clang-format on

        // Get the operation name and print it.
        let ident = mlirOperationGetName(operation);
        let identStr = mlirIdentifierStr(ident);
        eprint!("Operation name: '");
        for i in 0..identStr.length {
            eprint!("{}", *identStr.data.offset(i as isize) as u8 as char);
        }
        eprintln!("'");
        // CHECK: Operation name: 'arith.constant'

        // Get the identifier again and verify equal.
        let identAgain = mlirIdentifierGet(ctx, identStr);
        eprintln!(
            "Identifier equal: {}",
            mlirIdentifierEqual(ident, identAgain)
        );
        // CHECK: Identifier equal: 1

        // Get the block terminator and print it.
        let terminator = mlirBlockGetTerminator(block);
        eprint!("Terminator: ");
        mlirOperationPrint(
            terminator,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Terminator: func.return

        // Get the attribute by name.
        let hasValueAttr = mlirOperationHasInherentAttributeByName(
            operation,
            mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
        );
        if 0 != hasValueAttr {
            // CHECK: Has attr "value"
            eprint!("Has attr \"value\"");
        }

        let valueAttr0 = mlirOperationGetInherentAttributeByName(
            operation,
            mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
        );
        eprint!("Get attr \"value\": ");
        mlirAttributePrint(
            valueAttr0,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Get attr "value": 0 : index

        // Get a non-existing attribute and assert that it is null (sanity).
        eprintln!(
            "does_not_exist is null: {}",
            (std::ptr::null()
                == mlirOperationGetDiscardableAttributeByName(
                    operation,
                    mlirStringRefCreateFromCString("does_not_exist\0".as_ptr() as *const i8)
                )
                .ptr) as i8
        );
        // CHECK: does_not_exist is null: 1

        // Get result 0 and its type.
        let value = mlirOperationGetResult(operation, 0);
        eprint!("Result 0: ");
        mlirValuePrint(
            value,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        eprintln!("Value is null: {}", (std::ptr::null() == value.ptr) as i8);
        // CHECK: Result 0: {{.*}} = arith.constant 0 : index
        // CHECK: Value is null: 0

        let r#type = mlirValueGetType(value);
        eprint!("Result 0 type: ");
        mlirTypePrint(
            r#type,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Result 0 type: index

        // Set a discardable attribute.
        mlirOperationSetDiscardableAttributeByName(
            operation,
            mlirStringRefCreateFromCString("custom_attr\0".as_ptr() as *const i8),
            mlirBoolAttrGet(ctx, 1),
        );
        eprint!("Op with set attr: ");
        mlirOperationPrint(
            operation,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Op with set attr: {{.*}} {custom_attr = true}

        // Remove the attribute.
        eprintln!(
            "Remove attr: {}",
            mlirOperationRemoveDiscardableAttributeByName(
                operation,
                mlirStringRefCreateFromCString("custom_attr\0".as_ptr() as *const i8)
            )
        );
        eprintln!(
            "Remove attr again: {}",
            mlirOperationRemoveDiscardableAttributeByName(
                operation,
                mlirStringRefCreateFromCString("custom_attr\0".as_ptr() as *const i8)
            )
        );
        eprintln!(
            "Removed attr is null: {}",
            (std::ptr::null()
                == mlirOperationGetDiscardableAttributeByName(
                    operation,
                    mlirStringRefCreateFromCString("custom_attr\0".as_ptr() as *const i8)
                )
                .ptr) as i8
        );
        // CHECK: Remove attr: 1
        // CHECK: Remove attr again: 0
        // CHECK: Removed attr is null: 1

        // Add a large attribute to verify printing flags.
        let eltsShape = [4];
        let eltsData = [1, 2, 3, 4];
        mlirOperationSetDiscardableAttributeByName(
            operation,
            mlirStringRefCreateFromCString("elts\0".as_ptr() as *const i8),
            mlirDenseElementsAttrInt32Get(
                mlirRankedTensorTypeGet(
                    1,
                    eltsShape.as_ptr(),
                    mlirIntegerTypeGet(ctx, 32),
                    mlirAttributeGetNull(),
                ),
                4,
                eltsData.as_ptr(),
            ),
        );
        let mut flags = mlirOpPrintingFlagsCreate();
        mlirOpPrintingFlagsElideLargeElementsAttrs(flags, 2);
        mlirOpPrintingFlagsPrintGenericOpForm(flags);
        mlirOpPrintingFlagsEnableDebugInfo(flags, /*enable=*/ 1, /*prettyForm=*/ 0);
        mlirOpPrintingFlagsUseLocalScope(flags);
        eprint!("Op print with all flags: ");
        mlirOperationPrintWithFlags(
            operation,
            flags,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        eprint!("Op print with state: ");
        let state = mlirAsmStateCreateForOperation(parentOperation, flags);
        mlirOperationPrintWithState(
            operation,
            state,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // clang-format off
        // CHECK: Op print with all flags: %{{.*}} = "arith.constant"() <{value = 0 : index}> {elts = dense_resource<__elided__> : tensor<4xi32>} : () -> index loc(unknown)
        // clang-format on

        mlirOpPrintingFlagsDestroy(flags);
        flags = mlirOpPrintingFlagsCreate();
        mlirOpPrintingFlagsSkipRegions(flags);
        eprint!("Op print with skip regions flag: ");
        mlirOperationPrintWithFlags(
            function,
            flags,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // clang-format off
        // CHECK: Op print with skip regions flag: func.func @add(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>)
        // CHECK-NOT: constant
        // CHECK-NOT: return
        // clang-format on

        eprint!("With state: |");
        mlirValuePrintAsOperand(
            value,
            state,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        // CHECK: With state: |%0|
        eprintln!("|");
        mlirAsmStateDestroy(state);

        mlirOpPrintingFlagsDestroy(flags);
    }
}

/// Creates an operation with a region containing multiple blocks with
/// operations and dumps it. The blocks and operations are inserted using
/// block/operation-relative API and their final order is checked.
fn buildWithInsertionsAndPrint(ctx: MlirContext) {
    unsafe {
        let loc = mlirLocationUnknownGet(ctx);
        mlirContextSetAllowUnregisteredDialects(ctx, 1);

        let mut owningRegion = mlirRegionCreate();
        let nullBlock = mlirRegionGetFirstBlock(owningRegion);
        let mut state = mlirOperationStateGet(
            mlirStringRefCreateFromCString("insertion.order.test\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddOwnedRegions(&mut state, 1, &mut owningRegion);
        let op = mlirOperationCreate(&mut state);
        let region = mlirOperationGetRegion(op, 0);

        // Use integer types of different bitwidth as block arguments in order to
        // differentiate blocks.
        let i1 = mlirIntegerTypeGet(ctx, 1);
        let i2 = mlirIntegerTypeGet(ctx, 2);
        let i3 = mlirIntegerTypeGet(ctx, 3);
        let i4 = mlirIntegerTypeGet(ctx, 4);
        let i5 = mlirIntegerTypeGet(ctx, 5);
        let block1 = mlirBlockCreate(1, &i1, &loc);
        let block2 = mlirBlockCreate(1, &i2, &loc);
        let block3 = mlirBlockCreate(1, &i3, &loc);
        let block4 = mlirBlockCreate(1, &i4, &loc);
        let block5 = mlirBlockCreate(1, &i5, &loc);
        // Insert blocks so as to obtain the 1-2-3-4 order,
        mlirRegionInsertOwnedBlockBefore(region, nullBlock, block3);
        mlirRegionInsertOwnedBlockBefore(region, block3, block2);
        mlirRegionInsertOwnedBlockAfter(region, nullBlock, block1);
        mlirRegionInsertOwnedBlockAfter(region, block3, block4);
        mlirRegionInsertOwnedBlockBefore(region, block3, block5);

        let mut op1State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op1\0".as_ptr() as *const i8),
            loc,
        );
        let mut op2State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op2\0".as_ptr() as *const i8),
            loc,
        );
        let mut op3State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op3\0".as_ptr() as *const i8),
            loc,
        );
        let mut op4State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op4\0".as_ptr() as *const i8),
            loc,
        );
        let mut op5State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op5\0".as_ptr() as *const i8),
            loc,
        );
        let mut op6State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op6\0".as_ptr() as *const i8),
            loc,
        );
        let mut op7State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op7\0".as_ptr() as *const i8),
            loc,
        );
        let mut op8State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op8\0".as_ptr() as *const i8),
            loc,
        );
        let op1 = mlirOperationCreate(&mut op1State);
        let op2 = mlirOperationCreate(&mut op2State);
        let op3 = mlirOperationCreate(&mut op3State);
        let op4 = mlirOperationCreate(&mut op4State);
        let op5 = mlirOperationCreate(&mut op5State);
        let op6 = mlirOperationCreate(&mut op6State);
        let op7 = mlirOperationCreate(&mut op7State);
        let op8 = mlirOperationCreate(&mut op8State);

        // Insert operations in the first block so as to obtain the 1-2-3-4 order.
        let nullOperation = mlirBlockGetFirstOperation(block1);
        assert!(std::ptr::null() == nullOperation.ptr);
        mlirBlockInsertOwnedOperationBefore(block1, nullOperation, op3);
        mlirBlockInsertOwnedOperationBefore(block1, op3, op2);
        mlirBlockInsertOwnedOperationAfter(block1, nullOperation, op1);
        mlirBlockInsertOwnedOperationAfter(block1, op3, op4);

        // Append operations to the rest of blocks to make them non-empty and thus
        // printable.
        mlirBlockAppendOwnedOperation(block2, op5);
        mlirBlockAppendOwnedOperation(block3, op6);
        mlirBlockAppendOwnedOperation(block4, op7);
        mlirBlockAppendOwnedOperation(block5, op8);

        // Remove block5.
        mlirBlockDetach(block5);
        mlirBlockDestroy(block5);

        mlirOperationDump(op);
        mlirOperationDestroy(op);
        mlirContextSetAllowUnregisteredDialects(ctx, 0);
        // clang-format off
        // CHECK-LABEL:  "insertion.order.test"
        // CHECK:      ^{{.*}}(%{{.*}}: i1
        // CHECK:        "dummy.op1"
        // CHECK-NEXT:   "dummy.op2"
        // CHECK-NEXT:   "dummy.op3"
        // CHECK-NEXT:   "dummy.op4"
        // CHECK:      ^{{.*}}(%{{.*}}: i2
        // CHECK:        "dummy.op5"
        // CHECK-NOT:  ^{{.*}}(%{{.*}}: i5
        // CHECK-NOT:    "dummy.op8"
        // CHECK:      ^{{.*}}(%{{.*}}: i3
        // CHECK:        "dummy.op6"
        // CHECK:      ^{{.*}}(%{{.*}}: i4
        // CHECK:        "dummy.op7"
        // clang-format on
    }
}

/// Creates operations with type inference and tests various failure modes.
fn createOperationWithTypeInference(ctx: MlirContext) -> i32 {
    unsafe {
        let loc = mlirLocationUnknownGet(ctx);
        let iAttr = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), 4);

        // The shape.const_size op implements result type inference and is only used
        // for that reason.
        let mut state = mlirOperationStateGet(
            mlirStringRefCreateFromCString("shape.const_size\0".as_ptr() as *const i8),
            loc,
        );
        let valueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            iAttr,
        );
        mlirOperationStateAddAttributes(&mut state, 1, &valueAttr);
        mlirOperationStateEnableResultTypeInference(&mut state);

        // Expect result type inference to succeed.
        let op = mlirOperationCreate(&mut state);
        if std::ptr::null() == op.ptr {
            eprint!("ERROR: Result type inference unexpectedly failed");
            return 1;
        }

        // CHECK: RESULT_TYPE_INFERENCE: !shape.size
        eprint!("RESULT_TYPE_INFERENCE: ");
        mlirTypeDump(mlirValueGetType(mlirOperationGetResult(op, 0)));
        eprintln!();
        mlirOperationDestroy(op);
        0
    }
}

fn mlirStringRefCreate(ptr: *const i8, size: u64) -> StructMlirStringRef {
    StructMlirStringRef {
        data: ptr,
        length: size,
    }
}

fn mlirAttributeIsNull(attr: StructMlirAttribute) -> bool {
    std::ptr::null() == attr.ptr
}

struct ResourceDeleteUserData {
    pub name: *const u8,
}
unsafe impl Send for ResourceDeleteUserData {}
unsafe impl Sync for ResourceDeleteUserData {}

pub unsafe extern "C" fn reportResourceDelete(
    userData: *mut u8,
    _data: *const u8,
    _size: u64,
    _align: u64,
) -> u8 {
    let name = (*(userData as *const ResourceDeleteUserData)).name;
    let len = libc::strlen(name as *const i8);
    let name_slice = std::slice::from_raw_parts(name, len);
    let name_str = std::str::from_utf8_unchecked(name_slice);
    eprintln!("reportResourceDelete: {}", name_str);
    0
}

fn stringIsEqual(lhs: *const i8, rhs: MlirStringRef) -> bool {
    unsafe {
        if libc::strlen(lhs) as u64 != rhs.length {
            return false;
        }
        0 == libc::strncmp(lhs, rhs.data, rhs.length as usize)
    }
}

/// Dumps instances of all builtin types to check that C API works correctly.
/// Additionally, performs simple identity checks that a builtin type
/// constructed with C API can be inspected and has the expected type. The
/// latter achieves full coverage of C API for builtin types. Returns 0 on
/// success and a non-zero error code on failure.
fn printBuiltinTypes(ctx: MlirContext) -> i32 {
    unsafe {
        // Integer types.
        let i32 = mlirIntegerTypeGet(ctx, 32);
        let si32 = mlirIntegerTypeSignedGet(ctx, 32);
        let ui32 = mlirIntegerTypeUnsignedGet(ctx, 32);
        if 0 == mlirTypeIsAInteger(i32) || 0 != mlirTypeIsAF32(i32) {
            return 1;
        }
        if 0 == mlirTypeIsAInteger(si32) || 0 == mlirIntegerTypeIsSigned(si32) {
            return 2;
        }
        if 0 == mlirTypeIsAInteger(ui32) || 0 == mlirIntegerTypeIsUnsigned(ui32) {
            return 3;
        }
        if 0 != mlirTypeEqual(i32, ui32) || 0 != mlirTypeEqual(i32, si32) {
            return 4;
        }
        if mlirIntegerTypeGetWidth(i32) != mlirIntegerTypeGetWidth(si32) {
            return 5;
        }
        eprintln!("@types");
        mlirTypeDump(i32);
        eprintln!();
        mlirTypeDump(si32);
        eprintln!();
        mlirTypeDump(ui32);
        eprintln!();
        // CHECK-LABEL: @types
        // CHECK: i32
        // CHECK: si32
        // CHECK: ui32

        // Index type.
        let index = mlirIndexTypeGet(ctx);
        if 0 == mlirTypeIsAIndex(index) {
            return 6;
        }
        mlirTypeDump(index);
        eprintln!();
        // CHECK: index

        // Floating-point types.
        let bf16 = mlirBF16TypeGet(ctx);
        let f16 = mlirF16TypeGet(ctx);
        let f32 = mlirF32TypeGet(ctx);
        let f64 = mlirF64TypeGet(ctx);
        if 0 == mlirTypeIsABF16(bf16) {
            return 7;
        }
        if 0 == mlirTypeIsAF16(f16) {
            return 9;
        }
        if 0 == mlirTypeIsAF32(f32) {
            return 10;
        }
        if 0 == mlirTypeIsAF64(f64) {
            return 11;
        }
        mlirTypeDump(bf16);
        eprintln!();
        mlirTypeDump(f16);
        eprintln!();
        mlirTypeDump(f32);
        eprintln!();
        mlirTypeDump(f64);
        eprintln!();
        // CHECK: bf16
        // CHECK: f16
        // CHECK: f32
        // CHECK: f64

        // None type.
        let none = mlirNoneTypeGet(ctx);
        if 0 == mlirTypeIsANone(none) {
            return 12;
        }
        mlirTypeDump(none);
        eprintln!();
        // CHECK: none

        // Complex type.
        let cplx = mlirComplexTypeGet(f32);
        if 0 == mlirTypeIsAComplex(cplx)
            || 0 == mlirTypeEqual(mlirComplexTypeGetElementType(cplx), f32)
        {
            return 13;
        }
        mlirTypeDump(cplx);
        eprintln!();
        // CHECK: complex<f32>

        // Vector (and Shaped) type. ShapedType is a common base class for vectors,
        // memrefs and tensors, one cannot create instances of this class so it is
        // tested on an instance of vector type.
        let shape = [2_i64, 3_i64];
        let vector = mlirVectorTypeGet(shape.len() as i64, shape.as_ptr() as *const i64, f32);
        if 0 == mlirTypeIsAVector(vector) || 0 == mlirTypeIsAShaped(vector) {
            return 14;
        }
        if 0 == mlirTypeEqual(mlirShapedTypeGetElementType(vector), f32)
            || 0 == mlirShapedTypeHasRank(vector)
            || mlirShapedTypeGetRank(vector) != 2
            || mlirShapedTypeGetDimSize(vector, 0) != 2
            || 0 != mlirShapedTypeIsDynamicDim(vector, 0)
            || mlirShapedTypeGetDimSize(vector, 1) != 3
            || 0 == mlirShapedTypeHasStaticShape(vector)
        {
            return 15;
        }
        mlirTypeDump(vector);
        eprintln!();
        // CHECK: vector<2x3xf32>

        // Scalable vector type.
        let scalable = [0_u8, 1_u8];
        let scalableVector = mlirVectorTypeGetScalable(
            shape.len() as i64,
            shape.as_ptr() as *const i64,
            scalable.as_ptr() as *const u8,
            f32,
        );
        if 0 == mlirTypeIsAVector(scalableVector) {
            return 16;
        }
        if 0 == mlirVectorTypeIsScalable(scalableVector)
            || 0 != mlirVectorTypeIsDimScalable(scalableVector, 0)
            || 0 == mlirVectorTypeIsDimScalable(scalableVector, 1)
        {
            return 17;
        }
        mlirTypeDump(scalableVector);
        eprintln!();
        // CHECK: vector<2x[3]xf32>

        // Ranked tensor type.
        let rankedTensor = mlirRankedTensorTypeGet(
            shape.len() as i64,
            shape.as_ptr() as *const i64,
            f32,
            mlirAttributeGetNull(),
        );
        if 0 == mlirTypeIsATensor(rankedTensor)
            || 0 == mlirTypeIsARankedTensor(rankedTensor)
            || !mlirAttributeIsNull(mlirRankedTensorTypeGetEncoding(rankedTensor))
        {
            return 18;
        }
        mlirTypeDump(rankedTensor);
        eprintln!();
        // CHECK: tensor<2x3xf32>

        // Unranked tensor type.
        let unrankedTensor = mlirUnrankedTensorTypeGet(f32);
        if 0 == mlirTypeIsATensor(unrankedTensor)
            || 0 == mlirTypeIsAUnrankedTensor(unrankedTensor)
            || 0 != mlirShapedTypeHasRank(unrankedTensor)
        {
            return 19;
        }
        mlirTypeDump(unrankedTensor);
        eprintln!();
        // CHECK: tensor<*xf32>

        // MemRef type.
        let memSpace2 = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), 2);
        let memRef = mlirMemRefTypeContiguousGet(
            f32,
            shape.len() as i64,
            shape.as_ptr() as *const i64,
            memSpace2,
        );
        if 0 == mlirTypeIsAMemRef(memRef)
            || 0 == mlirAttributeEqual(mlirMemRefTypeGetMemorySpace(memRef), memSpace2)
        {
            return 20;
        }
        mlirTypeDump(memRef);
        eprintln!();
        // CHECK: memref<2x3xf32, 2>

        // Unranked MemRef type.
        let memSpace4 = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 64), 4);
        let unrankedMemRef = mlirUnrankedMemRefTypeGet(f32, memSpace4);
        if 0 == mlirTypeIsAUnrankedMemRef(unrankedMemRef)
            || 0 != mlirTypeIsAMemRef(unrankedMemRef)
            || 0 == mlirAttributeEqual(mlirUnrankedMemrefGetMemorySpace(unrankedMemRef), memSpace4)
        {
            return 21;
        }
        mlirTypeDump(unrankedMemRef);
        eprintln!();
        // CHECK: memref<*xf32, 4>

        // Tuple type.
        let types = [unrankedMemRef, f32];
        let tuple = mlirTupleTypeGet(ctx, 2, types.as_ptr() as *const StructMlirType);
        if 0 == mlirTypeIsATuple(tuple)
            || mlirTupleTypeGetNumTypes(tuple) != 2
            || 0 == mlirTypeEqual(mlirTupleTypeGetType(tuple, 0), unrankedMemRef)
            || 0 == mlirTypeEqual(mlirTupleTypeGetType(tuple, 1), f32)
        {
            return 22;
        }
        mlirTypeDump(tuple);
        eprintln!();
        // CHECK: tuple<memref<*xf32, 4>, f32>

        // Function type.
        let funcInputs = [mlirIndexTypeGet(ctx), mlirIntegerTypeGet(ctx, 1)];
        let funcResults = [
            mlirIntegerTypeGet(ctx, 16),
            mlirIntegerTypeGet(ctx, 32),
            mlirIntegerTypeGet(ctx, 64),
        ];
        let funcType = mlirFunctionTypeGet(
            ctx,
            2,
            funcInputs.as_ptr() as *const StructMlirType,
            3,
            funcResults.as_ptr() as *const StructMlirType,
        );
        if mlirFunctionTypeGetNumInputs(funcType) != 2 {
            return 23;
        }
        if mlirFunctionTypeGetNumResults(funcType) != 3 {
            return 24;
        }
        if 0 == mlirTypeEqual(funcInputs[0], mlirFunctionTypeGetInput(funcType, 0))
            || 0 == mlirTypeEqual(funcInputs[1], mlirFunctionTypeGetInput(funcType, 1))
        {
            return 25;
        }
        if 0 == mlirTypeEqual(funcResults[0], mlirFunctionTypeGetResult(funcType, 0))
            || 0 == mlirTypeEqual(funcResults[1], mlirFunctionTypeGetResult(funcType, 1))
            || 0 == mlirTypeEqual(funcResults[2], mlirFunctionTypeGetResult(funcType, 2))
        {
            return 26;
        }
        mlirTypeDump(funcType);
        eprintln!();
        // CHECK: (index, i1) -> (i16, i32, i64)

        // Opaque type.
        let namespace = mlirStringRefCreate("dialect\0".as_ptr() as *const i8, 7);
        let data = mlirStringRefCreate("type\0".as_ptr() as *const i8, 4);
        mlirContextSetAllowUnregisteredDialects(ctx, 1);
        let opaque = mlirOpaqueTypeGet(ctx, namespace, data);
        mlirContextSetAllowUnregisteredDialects(ctx, 0);
        if 0 == mlirTypeIsAOpaque(opaque)
            || 0 == mlirStringRefEqual(mlirOpaqueTypeGetDialectNamespace(opaque), namespace)
            || 0 == mlirStringRefEqual(mlirOpaqueTypeGetData(opaque), data)
        {
            return 27;
        }
        mlirTypeDump(opaque);
        eprintln!();
        // CHECK: !dialect.type

        0
    }
}

#[allow(non_upper_case_globals)]
static mut resourceI64BlobUserData: ResourceDeleteUserData = ResourceDeleteUserData {
    name: "resource_i64_blob\0".as_ptr() as *const u8,
};

fn printBuiltinAttributes(ctx: MlirContext) -> i32 {
    unsafe {
        let floating = mlirFloatAttrDoubleGet(ctx, mlirF64TypeGet(ctx), 2.0);
        if 0 == mlirAttributeIsAFloat(floating)
            || libm::fabs(mlirFloatAttrGetValueDouble(floating) - 2.0) > 1E-6
        {
            return 1;
        }
        eprintln!("@attrs");
        mlirAttributeDump(floating);
        // CHECK-LABEL: @attrs
        // CHECK: 2.000000e+00 : f64

        // Exercise mlirAttributeGetType() just for the first one.
        let floatingType = mlirAttributeGetType(floating);
        mlirTypeDump(floatingType);
        // CHECK: f64

        let integer = mlirIntegerAttrGet(mlirIntegerTypeGet(ctx, 32), 42);
        let signedInteger = mlirIntegerAttrGet(mlirIntegerTypeSignedGet(ctx, 8), -1);
        let unsignedInteger = mlirIntegerAttrGet(mlirIntegerTypeUnsignedGet(ctx, 8), 255);
        if 0 == mlirAttributeIsAInteger(integer)
            || mlirIntegerAttrGetValueInt(integer) != 42
            || mlirIntegerAttrGetValueSInt(signedInteger) != -1
            || mlirIntegerAttrGetValueUInt(unsignedInteger) != 255
        {
            return 2;
        }
        mlirAttributeDump(integer);
        mlirAttributeDump(signedInteger);
        mlirAttributeDump(unsignedInteger);
        // CHECK: 42 : i32
        // CHECK: -1 : si8
        // CHECK: 255 : ui8

        let boolean = mlirBoolAttrGet(ctx, 1);
        if 0 == mlirAttributeIsABool(boolean) || 0 == mlirBoolAttrGetValue(boolean) {
            return 3;
        }
        mlirAttributeDump(boolean);
        // CHECK: true

        let data = "abcdefghijklmnopqestuvwxyz\0";
        let opaque = mlirOpaqueAttrGet(
            ctx,
            mlirStringRefCreateFromCString("func\0".as_ptr() as *const i8),
            3,
            data.as_ptr() as *const i8,
            mlirNoneTypeGet(ctx),
        );
        if 0 == mlirAttributeIsAOpaque(opaque)
            || !stringIsEqual(
                "func\0".as_ptr() as *const i8,
                mlirOpaqueAttrGetDialectNamespace(opaque),
            )
        {
            return 4;
        }

        let opaqueData = mlirOpaqueAttrGetData(opaque);
        if opaqueData.length != 3
            || 0 != libc::strncmp(
                data.as_ptr() as *const i8,
                opaqueData.data,
                opaqueData.length as usize,
            )
        {
            return 5;
        }
        mlirAttributeDump(opaque);
        // CHECK: #func.abc

        let string = mlirStringAttrGet(
            ctx,
            mlirStringRefCreate(data.as_ptr().add(3) as *const i8, 2),
        );
        if 0 == mlirAttributeIsAString(string) {
            return 6;
        }

        let stringValue = mlirStringAttrGetValue(string);
        if stringValue.length != 2
            || 0 != libc::strncmp(
                data.as_ptr().add(3) as *const i8,
                stringValue.data,
                stringValue.length as usize,
            )
        {
            return 7;
        }
        mlirAttributeDump(string);
        // CHECK: "de"

        let flatSymbolRef = mlirFlatSymbolRefAttrGet(
            ctx,
            mlirStringRefCreate(data.as_ptr().add(5) as *const i8, 3),
        );
        if 0 == mlirAttributeIsAFlatSymbolRef(flatSymbolRef) {
            return 8;
        }

        let flatSymbolRefValue = mlirFlatSymbolRefAttrGetValue(flatSymbolRef);
        if flatSymbolRefValue.length != 3
            || 0 != libc::strncmp(
                data.as_ptr().add(5) as *const i8,
                flatSymbolRefValue.data,
                flatSymbolRefValue.length as usize,
            )
        {
            return 9;
        }
        mlirAttributeDump(flatSymbolRef);
        // CHECK: @fgh

        let symbols = [flatSymbolRef, flatSymbolRef];
        let symbolRef = mlirSymbolRefAttrGet(
            ctx,
            mlirStringRefCreate(data.as_ptr().add(8) as *const i8, 2),
            2,
            symbols.as_ptr(),
        );
        if 0 == mlirAttributeIsASymbolRef(symbolRef)
            || mlirSymbolRefAttrGetNumNestedReferences(symbolRef) != 2
            || 0 == mlirAttributeEqual(
                mlirSymbolRefAttrGetNestedReference(symbolRef, 0),
                flatSymbolRef,
            )
            || 0 == mlirAttributeEqual(
                mlirSymbolRefAttrGetNestedReference(symbolRef, 1),
                flatSymbolRef,
            )
        {
            return 10;
        }

        let symbolRefLeaf = mlirSymbolRefAttrGetLeafReference(symbolRef);
        let symbolRefRoot = mlirSymbolRefAttrGetRootReference(symbolRef);
        if symbolRefLeaf.length != 3
            || 0 != libc::strncmp(
                data.as_ptr().add(5) as *const i8,
                symbolRefLeaf.data,
                symbolRefLeaf.length as usize,
            )
            || symbolRefRoot.length != 2
            || 0 != libc::strncmp(
                data.as_ptr().add(8) as *const i8,
                symbolRefRoot.data,
                symbolRefRoot.length as usize,
            )
        {
            return 11;
        }
        mlirAttributeDump(symbolRef);
        // CHECK: @ij::@fgh::@fgh

        let r#type = mlirTypeAttrGet(mlirF32TypeGet(ctx));
        if 0 == mlirAttributeIsAType(r#type)
            || 0 == mlirTypeEqual(mlirF32TypeGet(ctx), mlirTypeAttrGetValue(r#type))
        {
            return 12;
        }
        mlirAttributeDump(r#type);
        // CHECK: f32

        let unit = mlirUnitAttrGet(ctx);
        if 0 == mlirAttributeIsAUnit(unit) {
            return 13;
        }
        mlirAttributeDump(unit);
        // CHECK: unit

        let shape = [1_i64, 2_i64];

        let bools = [0 as std::ffi::c_int, 1 as std::ffi::c_int];
        let uints8 = [0_u8, 1_u8];
        let ints8 = [0_i8, 1_i8];
        let uints16 = [0_u16, 1_u16];
        let ints16 = [0_i16, 1_i16];
        let uints32 = [0_u32, 1_u32];
        let ints32 = [0_i32, 1_i32];
        let mut uints64 = [0_u64, 1_u64];
        let ints64 = [0_i64, 1_i64];
        let floats = [0.0_f32, 1.0_f32];
        let doubles = [0.0_f64, 1.0_f64];
        let bf16s = [0x0_u16, 0x3f80_u16];
        let f16s = [0x0_u16, 0x3c00_u16];
        let encoding = mlirAttributeGetNull();
        let boolElements = mlirDenseElementsAttrBoolGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 1), encoding),
            2,
            bools.as_ptr(),
        );
        let uint8Elements = mlirDenseElementsAttrUInt8Get(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 8),
                encoding,
            ),
            2,
            uints8.as_ptr(),
        );
        let int8Elements = mlirDenseElementsAttrInt8Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 8), encoding),
            2,
            ints8.as_ptr(),
        );
        let uint16Elements = mlirDenseElementsAttrUInt16Get(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 16),
                encoding,
            ),
            2,
            uints16.as_ptr(),
        );
        let int16Elements = mlirDenseElementsAttrInt16Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 16), encoding),
            2,
            ints16.as_ptr(),
        );
        let uint32Elements = mlirDenseElementsAttrUInt32Get(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 32),
                encoding,
            ),
            2,
            uints32.as_ptr(),
        );
        let int32Elements = mlirDenseElementsAttrInt32Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 32), encoding),
            2,
            ints32.as_ptr(),
        );
        let uint64Elements = mlirDenseElementsAttrUInt64Get(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 64),
                encoding,
            ),
            2,
            uints64.as_ptr(),
        );
        let int64Elements = mlirDenseElementsAttrInt64Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 64), encoding),
            2,
            ints64.as_ptr(),
        );
        let floatElements = mlirDenseElementsAttrFloatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF32TypeGet(ctx), encoding),
            2,
            floats.as_ptr(),
        );
        let doubleElements = mlirDenseElementsAttrDoubleGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF64TypeGet(ctx), encoding),
            2,
            doubles.as_ptr(),
        );
        let bf16Elements = mlirDenseElementsAttrBFloat16Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirBF16TypeGet(ctx), encoding),
            2,
            bf16s.as_ptr(),
        );
        let f16Elements = mlirDenseElementsAttrFloat16Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF16TypeGet(ctx), encoding),
            2,
            f16s.as_ptr(),
        );

        if 0 == mlirAttributeIsADenseElements(boolElements)
            || 0 == mlirAttributeIsADenseElements(uint8Elements)
            || 0 == mlirAttributeIsADenseElements(int8Elements)
            || 0 == mlirAttributeIsADenseElements(uint32Elements)
            || 0 == mlirAttributeIsADenseElements(int32Elements)
            || 0 == mlirAttributeIsADenseElements(uint64Elements)
            || 0 == mlirAttributeIsADenseElements(int64Elements)
            || 0 == mlirAttributeIsADenseElements(floatElements)
            || 0 == mlirAttributeIsADenseElements(doubleElements)
            || 0 == mlirAttributeIsADenseElements(bf16Elements)
            || 0 == mlirAttributeIsADenseElements(f16Elements)
        {
            return 14;
        }

        if mlirDenseElementsAttrGetBoolValue(boolElements, 1) != 1
            || mlirDenseElementsAttrGetUInt8Value(uint8Elements, 1) != 1
            || mlirDenseElementsAttrGetInt8Value(int8Elements, 1) != 1
            || mlirDenseElementsAttrGetUInt16Value(uint16Elements, 1) != 1
            || mlirDenseElementsAttrGetInt16Value(int16Elements, 1) != 1
            || mlirDenseElementsAttrGetUInt32Value(uint32Elements, 1) != 1
            || mlirDenseElementsAttrGetInt32Value(int32Elements, 1) != 1
            || mlirDenseElementsAttrGetUInt64Value(uint64Elements, 1) != 1
            || mlirDenseElementsAttrGetInt64Value(int64Elements, 1) != 1
            || libm::fabsf(mlirDenseElementsAttrGetFloatValue(floatElements, 1) - 1.0f32) > 1E-6f32
            || libm::fabs(mlirDenseElementsAttrGetDoubleValue(doubleElements, 1) - 1.0) > 1E-6
        {
            return 15;
        }

        mlirAttributeDump(boolElements);
        mlirAttributeDump(uint8Elements);
        mlirAttributeDump(int8Elements);
        mlirAttributeDump(uint32Elements);
        mlirAttributeDump(int32Elements);
        mlirAttributeDump(uint64Elements);
        mlirAttributeDump(int64Elements);
        mlirAttributeDump(floatElements);
        mlirAttributeDump(doubleElements);
        mlirAttributeDump(bf16Elements);
        mlirAttributeDump(f16Elements);
        // CHECK: dense<{{\[}}[false, true]]> : tensor<1x2xi1>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui8>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi8>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui32>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi32>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xui64>
        // CHECK: dense<{{\[}}[0, 1]]> : tensor<1x2xi64>
        // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf32>
        // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf64>
        // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xbf16>
        // CHECK: dense<{{\[}}[0.000000e+00, 1.000000e+00]]> : tensor<1x2xf16>

        let splatBool = mlirDenseElementsAttrBoolSplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 1), encoding),
            1,
        );
        let splatUInt8 = mlirDenseElementsAttrUInt8SplatGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 8),
                encoding,
            ),
            1,
        );
        let splatInt8 = mlirDenseElementsAttrInt8SplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 8), encoding),
            1,
        );
        let splatUInt32 = mlirDenseElementsAttrUInt32SplatGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 32),
                encoding,
            ),
            1,
        );
        let splatInt32 = mlirDenseElementsAttrInt32SplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 32), encoding),
            1,
        );
        let splatUInt64 = mlirDenseElementsAttrUInt64SplatGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 64),
                encoding,
            ),
            1,
        );
        let splatInt64 = mlirDenseElementsAttrInt64SplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 64), encoding),
            1,
        );
        let splatFloat = mlirDenseElementsAttrFloatSplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF32TypeGet(ctx), encoding),
            1.0f32,
        );
        let splatDouble = mlirDenseElementsAttrDoubleSplatGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF64TypeGet(ctx), encoding),
            1.0,
        );

        if 0 == mlirAttributeIsADenseElements(splatBool)
            || 0 == mlirDenseElementsAttrIsSplat(splatBool)
            || 0 == mlirAttributeIsADenseElements(splatUInt8)
            || 0 == mlirDenseElementsAttrIsSplat(splatUInt8)
            || 0 == mlirAttributeIsADenseElements(splatInt8)
            || 0 == mlirDenseElementsAttrIsSplat(splatInt8)
            || 0 == mlirAttributeIsADenseElements(splatUInt32)
            || 0 == mlirDenseElementsAttrIsSplat(splatUInt32)
            || 0 == mlirAttributeIsADenseElements(splatInt32)
            || 0 == mlirDenseElementsAttrIsSplat(splatInt32)
            || 0 == mlirAttributeIsADenseElements(splatUInt64)
            || 0 == mlirDenseElementsAttrIsSplat(splatUInt64)
            || 0 == mlirAttributeIsADenseElements(splatInt64)
            || 0 == mlirDenseElementsAttrIsSplat(splatInt64)
            || 0 == mlirAttributeIsADenseElements(splatFloat)
            || 0 == mlirDenseElementsAttrIsSplat(splatFloat)
            || 0 == mlirAttributeIsADenseElements(splatDouble)
            || 0 == mlirDenseElementsAttrIsSplat(splatDouble)
        {
            return 16;
        }

        if mlirDenseElementsAttrGetBoolSplatValue(splatBool) != 1
            || mlirDenseElementsAttrGetUInt8SplatValue(splatUInt8) != 1
            || mlirDenseElementsAttrGetInt8SplatValue(splatInt8) != 1
            || mlirDenseElementsAttrGetUInt32SplatValue(splatUInt32) != 1
            || mlirDenseElementsAttrGetInt32SplatValue(splatInt32) != 1
            || mlirDenseElementsAttrGetUInt64SplatValue(splatUInt64) != 1
            || mlirDenseElementsAttrGetInt64SplatValue(splatInt64) != 1
            || libm::fabsf(mlirDenseElementsAttrGetFloatSplatValue(splatFloat) - 1.0f32) > 1E-6f32
            || libm::fabs(mlirDenseElementsAttrGetDoubleSplatValue(splatDouble) - 1.0) > 1E-6
        {
            return 17;
        }

        let uint8RawData = mlirDenseElementsAttrGetRawData(uint8Elements) as *const u8;
        let int8RawData = mlirDenseElementsAttrGetRawData(int8Elements) as *const i8;
        let uint32RawData = mlirDenseElementsAttrGetRawData(uint32Elements) as *const u32;
        let int32RawData = mlirDenseElementsAttrGetRawData(int32Elements) as *const i32;
        let uint64RawData = mlirDenseElementsAttrGetRawData(uint64Elements) as *const u64;
        let int64RawData = mlirDenseElementsAttrGetRawData(int64Elements) as *const i64;
        let floatRawData = mlirDenseElementsAttrGetRawData(floatElements) as *const f32;
        let doubleRawData = mlirDenseElementsAttrGetRawData(doubleElements) as *const f64;
        let bf16RawData = mlirDenseElementsAttrGetRawData(bf16Elements) as *const i16;
        let f16RawData = mlirDenseElementsAttrGetRawData(f16Elements) as *const i16;
        if uint8RawData.read() != 0u8
            || uint8RawData.add(1).read() != 1u8
            || int8RawData.read() != 0
            || int8RawData.add(1).read() != 1
            || uint32RawData.read() != 0u32
            || uint32RawData.add(1).read() != 1u32
            || int32RawData.read() != 0
            || int32RawData.add(1).read() != 1
            || uint64RawData.read() != 0u64
            || uint64RawData.add(1).read() != 1u64
            || int64RawData.read() != 0
            || int64RawData.add(1).read() != 1
            || floatRawData.read() != 0.0f32
            || floatRawData.add(1).read() != 1.0f32
            || doubleRawData.read() != 0.0
            || doubleRawData.add(1).read() != 1.0
            || bf16RawData.read() != 0
            || bf16RawData.add(1).read() != 0x3f80
            || f16RawData.read() != 0
            || f16RawData.add(1).read() != 0x3c00
        {
            return 18;
        }

        mlirAttributeDump(splatBool);
        mlirAttributeDump(splatUInt8);
        mlirAttributeDump(splatInt8);
        mlirAttributeDump(splatUInt32);
        mlirAttributeDump(splatInt32);
        mlirAttributeDump(splatUInt64);
        mlirAttributeDump(splatInt64);
        mlirAttributeDump(splatFloat);
        mlirAttributeDump(splatDouble);
        // CHECK: dense<true> : tensor<1x2xi1>
        // CHECK: dense<1> : tensor<1x2xui8>
        // CHECK: dense<1> : tensor<1x2xi8>
        // CHECK: dense<1> : tensor<1x2xui32>
        // CHECK: dense<1> : tensor<1x2xi32>
        // CHECK: dense<1> : tensor<1x2xui64>
        // CHECK: dense<1> : tensor<1x2xi64>
        // CHECK: dense<1.000000e+00> : tensor<1x2xf32>
        // CHECK: dense<1.000000e+00> : tensor<1x2xf64>

        mlirAttributeDump(mlirElementsAttrGetValue(
            floatElements,
            2,
            uints64.as_mut_ptr(),
        ));
        mlirAttributeDump(mlirElementsAttrGetValue(
            doubleElements,
            2,
            uints64.as_mut_ptr(),
        ));
        mlirAttributeDump(mlirElementsAttrGetValue(
            bf16Elements,
            2,
            uints64.as_mut_ptr(),
        ));
        mlirAttributeDump(mlirElementsAttrGetValue(
            f16Elements,
            2,
            uints64.as_mut_ptr(),
        ));
        // CHECK: 1.000000e+00 : f32
        // CHECK: 1.000000e+00 : f64
        // CHECK: 1.000000e+00 : bf16
        // CHECK: 1.000000e+00 : f16

        let indices = [0, 1];
        let one = 1;
        let indicesAttr = mlirDenseElementsAttrInt64Get(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 64), encoding),
            2,
            indices.as_ptr(),
        );
        let valuesAttr = mlirDenseElementsAttrFloatGet(
            mlirRankedTensorTypeGet(1, &one, mlirF32TypeGet(ctx), encoding),
            1,
            floats.as_ptr(),
        );
        let sparseAttr = mlirSparseElementsAttribute(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF32TypeGet(ctx), encoding),
            indicesAttr,
            valuesAttr,
        );
        mlirAttributeDump(sparseAttr);
        // CHECK: sparse<{{\[}}[0, 1]], 0.000000e+00> : tensor<1x2xf32>

        let boolArray = mlirDenseBoolArrayGet(ctx, 2, bools.as_ptr());
        let int8Array = mlirDenseI8ArrayGet(ctx, 2, ints8.as_ptr());
        let int16Array = mlirDenseI16ArrayGet(ctx, 2, ints16.as_ptr());
        let int32Array = mlirDenseI32ArrayGet(ctx, 2, ints32.as_ptr());
        let int64Array = mlirDenseI64ArrayGet(ctx, 2, ints64.as_ptr());
        let floatArray = mlirDenseF32ArrayGet(ctx, 2, floats.as_ptr());
        let doubleArray = mlirDenseF64ArrayGet(ctx, 2, doubles.as_ptr());
        if 0 == mlirAttributeIsADenseBoolArray(boolArray)
            || 0 == mlirAttributeIsADenseI8Array(int8Array)
            || 0 == mlirAttributeIsADenseI16Array(int16Array)
            || 0 == mlirAttributeIsADenseI32Array(int32Array)
            || 0 == mlirAttributeIsADenseI64Array(int64Array)
            || 0 == mlirAttributeIsADenseF32Array(floatArray)
            || 0 == mlirAttributeIsADenseF64Array(doubleArray)
        {
            return 19;
        }

        if mlirDenseArrayGetNumElements(boolArray) != 2
            || mlirDenseArrayGetNumElements(int8Array) != 2
            || mlirDenseArrayGetNumElements(int16Array) != 2
            || mlirDenseArrayGetNumElements(int32Array) != 2
            || mlirDenseArrayGetNumElements(int64Array) != 2
            || mlirDenseArrayGetNumElements(floatArray) != 2
            || mlirDenseArrayGetNumElements(doubleArray) != 2
        {
            return 20;
        }

        if mlirDenseBoolArrayGetElement(boolArray, 1) != 1
            || mlirDenseI8ArrayGetElement(int8Array, 1) != 1
            || mlirDenseI16ArrayGetElement(int16Array, 1) != 1
            || mlirDenseI32ArrayGetElement(int32Array, 1) != 1
            || mlirDenseI64ArrayGetElement(int64Array, 1) != 1
            || libm::fabsf(mlirDenseF32ArrayGetElement(floatArray, 1) - 1.0f32) > 1E-6f32
            || libm::fabs(mlirDenseF64ArrayGetElement(doubleArray, 1) - 1.0) > 1E-6
        {
            return 21;
        }

        let layoutStrides = [5, 7, 13];
        let stridedLayoutAttr = mlirStridedLayoutAttrGet(ctx, 42, 3, &layoutStrides[0]);

        // CHECK: strided<[5, 7, 13], offset: 42>
        mlirAttributeDump(stridedLayoutAttr);

        if mlirStridedLayoutAttrGetOffset(stridedLayoutAttr) != 42
            || mlirStridedLayoutAttrGetNumStrides(stridedLayoutAttr) != 3
            || mlirStridedLayoutAttrGetStride(stridedLayoutAttr, 0) != 5
            || mlirStridedLayoutAttrGetStride(stridedLayoutAttr, 1) != 7
            || mlirStridedLayoutAttrGetStride(stridedLayoutAttr, 2) != 13
        {
            return 22;
        }
        let uint8Blob = mlirUnmanagedDenseUInt8ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 8),
                encoding,
            ),
            mlirStringRefCreateFromCString("resource_ui8\0".as_ptr() as *const i8),
            2,
            uints8.as_ptr(),
        );
        let uint16Blob = mlirUnmanagedDenseUInt16ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 16),
                encoding,
            ),
            mlirStringRefCreateFromCString("resource_ui16\0".as_ptr() as *const i8),
            2,
            uints16.as_ptr(),
        );
        let uint32Blob = mlirUnmanagedDenseUInt32ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 32),
                encoding,
            ),
            mlirStringRefCreateFromCString("resource_ui32\0".as_ptr() as *const i8),
            2,
            uints32.as_ptr(),
        );
        let uint64Blob = mlirUnmanagedDenseUInt64ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(
                2,
                shape.as_ptr(),
                mlirIntegerTypeUnsignedGet(ctx, 64),
                encoding,
            ),
            mlirStringRefCreateFromCString("resource_ui64\0".as_ptr() as *const i8),
            2,
            uints64.as_ptr(),
        );
        let int8Blob = mlirUnmanagedDenseInt8ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 8), encoding),
            mlirStringRefCreateFromCString("resource_i8\0".as_ptr() as *const i8),
            2,
            ints8.as_ptr(),
        );
        let int16Blob = mlirUnmanagedDenseInt16ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 16), encoding),
            mlirStringRefCreateFromCString("resource_i16\0".as_ptr() as *const i8),
            2,
            ints16.as_ptr(),
        );
        let int32Blob = mlirUnmanagedDenseInt32ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 32), encoding),
            mlirStringRefCreateFromCString("resource_i32\0".as_ptr() as *const i8),
            2,
            ints32.as_ptr(),
        );
        let int64Blob = mlirUnmanagedDenseInt64ResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 64), encoding),
            mlirStringRefCreateFromCString("resource_i64\0".as_ptr() as *const i8),
            2,
            ints64.as_ptr(),
        );
        let floatsBlob = mlirUnmanagedDenseFloatResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF32TypeGet(ctx), encoding),
            mlirStringRefCreateFromCString("resource_f32\0".as_ptr() as *const i8),
            2,
            floats.as_ptr(),
        );
        let doublesBlob = mlirUnmanagedDenseDoubleResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirF64TypeGet(ctx), encoding),
            mlirStringRefCreateFromCString("resource_f64\0".as_ptr() as *const i8),
            2,
            doubles.as_ptr(),
        );
        let userData = std::ptr::addr_of_mut!(resourceI64BlobUserData) as *mut _;
        let blobBlob = mlirUnmanagedDenseResourceElementsAttrGet(
            mlirRankedTensorTypeGet(2, shape.as_ptr(), mlirIntegerTypeGet(ctx, 64), encoding),
            mlirStringRefCreateFromCString("resource_i64_blob\0".as_ptr() as *const i8),
            /*data=*/ uints64.as_ptr() as *mut std::ffi::c_void,
            /*dataLength=*/ (std::mem::size_of::<u64>() * uints64.len()) as u64,
            /*dataAlignment=*/ std::mem::align_of::<u64>() as u64,
            /*dataIsMutable=*/ 0u8,
            /*deleter=*/
            reportResourceDelete
                as *mut extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, u64, u64),
            /*userData=*/ userData as *mut std::ffi::c_void,
        );

        mlirAttributeDump(uint8Blob);
        mlirAttributeDump(uint16Blob);
        mlirAttributeDump(uint32Blob);
        mlirAttributeDump(uint64Blob);
        mlirAttributeDump(int8Blob);
        mlirAttributeDump(int16Blob);
        mlirAttributeDump(int32Blob);
        mlirAttributeDump(int64Blob);
        mlirAttributeDump(floatsBlob);
        mlirAttributeDump(doublesBlob);
        mlirAttributeDump(blobBlob);
        // CHECK: dense_resource<resource_ui8> : tensor<1x2xui8>
        // CHECK: dense_resource<resource_ui16> : tensor<1x2xui16>
        // CHECK: dense_resource<resource_ui32> : tensor<1x2xui32>
        // CHECK: dense_resource<resource_ui64> : tensor<1x2xui64>
        // CHECK: dense_resource<resource_i8> : tensor<1x2xi8>
        // CHECK: dense_resource<resource_i16> : tensor<1x2xi16>
        // CHECK: dense_resource<resource_i32> : tensor<1x2xi32>
        // CHECK: dense_resource<resource_i64> : tensor<1x2xi64>
        // CHECK: dense_resource<resource_f32> : tensor<1x2xf32>
        // CHECK: dense_resource<resource_f64> : tensor<1x2xf64>
        // CHECK: dense_resource<resource_i64_blob> : tensor<1x2xi64>

        if mlirDenseUInt8ResourceElementsAttrGetValue(uint8Blob, 1) != 1
            || mlirDenseUInt16ResourceElementsAttrGetValue(uint16Blob, 1) != 1
            || mlirDenseUInt32ResourceElementsAttrGetValue(uint32Blob, 1) != 1
            || mlirDenseUInt64ResourceElementsAttrGetValue(uint64Blob, 1) != 1
            || mlirDenseInt8ResourceElementsAttrGetValue(int8Blob, 1) != 1
            || mlirDenseInt16ResourceElementsAttrGetValue(int16Blob, 1) != 1
            || mlirDenseInt32ResourceElementsAttrGetValue(int32Blob, 1) != 1
            || mlirDenseInt64ResourceElementsAttrGetValue(int64Blob, 1) != 1
            || libm::fabsf(mlirDenseF32ArrayGetElement(floatArray, 1) - 1.0f32) > 1E-6f32
            || libm::fabsf(mlirDenseFloatResourceElementsAttrGetValue(floatsBlob, 1) - 1.0f32)
                > 1e-6
            || libm::fabs(mlirDenseDoubleResourceElementsAttrGetValue(doublesBlob, 1) - 1.0f64)
                > 1e-6
            || mlirDenseUInt64ResourceElementsAttrGetValue(blobBlob, 1) != 1
        {
            return 23;
        }

        let loc = mlirLocationUnknownGet(ctx);
        let locAttr = mlirLocationGetAttribute(loc);
        if 0 == mlirAttributeIsALocation(locAttr) {
            return 24;
        }

        0
    }
}

fn printAffineMap(ctx: MlirContext) -> i32 {
    unsafe {
        let emptyAffineMap = mlirAffineMapEmptyGet(ctx);
        let affineMap = mlirAffineMapZeroResultGet(ctx, 3, 2);
        let constAffineMap = mlirAffineMapConstantGet(ctx, 2);
        let multiDimIdentityAffineMap = mlirAffineMapMultiDimIdentityGet(ctx, 3);
        let minorIdentityAffineMap = mlirAffineMapMinorIdentityGet(ctx, 3, 2);
        let mut permutation = [
            1 as std::ffi::c_uint,
            2 as std::ffi::c_uint,
            0 as std::ffi::c_uint,
        ];
        let permutationAffineMap =
            mlirAffineMapPermutationGet(ctx, permutation.len() as i64, permutation.as_mut_ptr());

        eprintln!("@affineMap");
        mlirAffineMapDump(emptyAffineMap);
        mlirAffineMapDump(affineMap);
        mlirAffineMapDump(constAffineMap);
        mlirAffineMapDump(multiDimIdentityAffineMap);
        mlirAffineMapDump(minorIdentityAffineMap);
        mlirAffineMapDump(permutationAffineMap);
        // CHECK-LABEL: @affineMap
        // CHECK: () -> ()
        // CHECK: (d0, d1, d2)[s0, s1] -> ()
        // CHECK: () -> (2)
        // CHECK: (d0, d1, d2) -> (d0, d1, d2)
        // CHECK: (d0, d1, d2) -> (d1, d2)
        // CHECK: (d0, d1, d2) -> (d1, d2, d0)

        if 0 == mlirAffineMapIsIdentity(emptyAffineMap)
            || 0 != mlirAffineMapIsIdentity(affineMap)
            || 0 != mlirAffineMapIsIdentity(constAffineMap)
            || 0 == mlirAffineMapIsIdentity(multiDimIdentityAffineMap)
            || 0 != mlirAffineMapIsIdentity(minorIdentityAffineMap)
            || 0 != mlirAffineMapIsIdentity(permutationAffineMap)
        {
            return 1;
        }

        if 0 == mlirAffineMapIsMinorIdentity(emptyAffineMap)
            || 0 != mlirAffineMapIsMinorIdentity(affineMap)
            || 0 == mlirAffineMapIsMinorIdentity(multiDimIdentityAffineMap)
            || 0 == mlirAffineMapIsMinorIdentity(minorIdentityAffineMap)
            || 0 != mlirAffineMapIsMinorIdentity(permutationAffineMap)
        {
            return 2;
        }

        if 0 == mlirAffineMapIsEmpty(emptyAffineMap)
            || 0 != mlirAffineMapIsEmpty(affineMap)
            || 0 != mlirAffineMapIsEmpty(constAffineMap)
            || 0 != mlirAffineMapIsEmpty(multiDimIdentityAffineMap)
            || 0 != mlirAffineMapIsEmpty(minorIdentityAffineMap)
            || 0 != mlirAffineMapIsEmpty(permutationAffineMap)
        {
            return 3;
        }

        if 0 != mlirAffineMapIsSingleConstant(emptyAffineMap)
            || 0 != mlirAffineMapIsSingleConstant(affineMap)
            || 0 == mlirAffineMapIsSingleConstant(constAffineMap)
            || 0 != mlirAffineMapIsSingleConstant(multiDimIdentityAffineMap)
            || 0 != mlirAffineMapIsSingleConstant(minorIdentityAffineMap)
            || 0 != mlirAffineMapIsSingleConstant(permutationAffineMap)
        {
            return 4;
        }

        if mlirAffineMapGetSingleConstantResult(constAffineMap) != 2 {
            return 5;
        }

        if mlirAffineMapGetNumDims(emptyAffineMap) != 0
            || mlirAffineMapGetNumDims(affineMap) != 3
            || mlirAffineMapGetNumDims(constAffineMap) != 0
            || mlirAffineMapGetNumDims(multiDimIdentityAffineMap) != 3
            || mlirAffineMapGetNumDims(minorIdentityAffineMap) != 3
            || mlirAffineMapGetNumDims(permutationAffineMap) != 3
        {
            return 6;
        }

        if mlirAffineMapGetNumSymbols(emptyAffineMap) != 0
            || mlirAffineMapGetNumSymbols(affineMap) != 2
            || mlirAffineMapGetNumSymbols(constAffineMap) != 0
            || mlirAffineMapGetNumSymbols(multiDimIdentityAffineMap) != 0
            || mlirAffineMapGetNumSymbols(minorIdentityAffineMap) != 0
            || mlirAffineMapGetNumSymbols(permutationAffineMap) != 0
        {
            return 7;
        }

        if mlirAffineMapGetNumResults(emptyAffineMap) != 0
            || mlirAffineMapGetNumResults(affineMap) != 0
            || mlirAffineMapGetNumResults(constAffineMap) != 1
            || mlirAffineMapGetNumResults(multiDimIdentityAffineMap) != 3
            || mlirAffineMapGetNumResults(minorIdentityAffineMap) != 2
            || mlirAffineMapGetNumResults(permutationAffineMap) != 3
        {
            return 8;
        }

        if mlirAffineMapGetNumInputs(emptyAffineMap) != 0
            || mlirAffineMapGetNumInputs(affineMap) != 5
            || mlirAffineMapGetNumInputs(constAffineMap) != 0
            || mlirAffineMapGetNumInputs(multiDimIdentityAffineMap) != 3
            || mlirAffineMapGetNumInputs(minorIdentityAffineMap) != 3
            || mlirAffineMapGetNumInputs(permutationAffineMap) != 3
        {
            return 9;
        }

        if 0 == mlirAffineMapIsProjectedPermutation(emptyAffineMap)
            || 0 == mlirAffineMapIsPermutation(emptyAffineMap)
            || 0 != mlirAffineMapIsProjectedPermutation(affineMap)
            || 0 != mlirAffineMapIsPermutation(affineMap)
            || 0 != mlirAffineMapIsProjectedPermutation(constAffineMap)
            || 0 != mlirAffineMapIsPermutation(constAffineMap)
            || 0 == mlirAffineMapIsProjectedPermutation(multiDimIdentityAffineMap)
            || 0 == mlirAffineMapIsPermutation(multiDimIdentityAffineMap)
            || 0 == mlirAffineMapIsProjectedPermutation(minorIdentityAffineMap)
            || 0 != mlirAffineMapIsPermutation(minorIdentityAffineMap)
            || 0 == mlirAffineMapIsProjectedPermutation(permutationAffineMap)
            || 0 == mlirAffineMapIsPermutation(permutationAffineMap)
        {
            return 10;
        }

        let mut sub = [1_isize];

        let subMap = mlirAffineMapGetSubMap(
            multiDimIdentityAffineMap,
            sub.len() as i64,
            sub.as_mut_ptr() as *mut i64,
        );
        let majorSubMap = mlirAffineMapGetMajorSubMap(multiDimIdentityAffineMap, 1);
        let minorSubMap = mlirAffineMapGetMinorSubMap(multiDimIdentityAffineMap, 1);

        mlirAffineMapDump(subMap);
        mlirAffineMapDump(majorSubMap);
        mlirAffineMapDump(minorSubMap);
        // CHECK: (d0, d1, d2) -> (d1)
        // CHECK: (d0, d1, d2) -> (d0)
        // CHECK: (d0, d1, d2) -> (d2)

        // CHECK: distinct[0]<"foo">
        mlirAttributeDump(mlirDisctinctAttrCreate(mlirStringAttrGet(
            ctx,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        )));

        0
    }
}

fn printAffineExpr(ctx: MlirContext) -> i32 {
    unsafe {
        let affineDimExpr = mlirAffineDimExprGet(ctx, 5);
        let affineSymbolExpr = mlirAffineSymbolExprGet(ctx, 5);
        let affineConstantExpr = mlirAffineConstantExprGet(ctx, 5);
        let affineAddExpr = mlirAffineAddExprGet(affineDimExpr, affineSymbolExpr);
        let affineMulExpr = mlirAffineMulExprGet(affineDimExpr, affineSymbolExpr);
        let affineModExpr = mlirAffineModExprGet(affineDimExpr, affineSymbolExpr);
        let affineFloorDivExpr = mlirAffineFloorDivExprGet(affineDimExpr, affineSymbolExpr);
        let affineCeilDivExpr = mlirAffineCeilDivExprGet(affineDimExpr, affineSymbolExpr);

        // Tests mlirAffineExprDump.
        eprintln!("@affineExpr");
        mlirAffineExprDump(affineDimExpr);
        mlirAffineExprDump(affineSymbolExpr);
        mlirAffineExprDump(affineConstantExpr);
        mlirAffineExprDump(affineAddExpr);
        mlirAffineExprDump(affineMulExpr);
        mlirAffineExprDump(affineModExpr);
        mlirAffineExprDump(affineFloorDivExpr);
        mlirAffineExprDump(affineCeilDivExpr);
        // CHECK-LABEL: @affineExpr
        // CHECK: d5
        // CHECK: s5
        // CHECK: 5
        // CHECK: d5 + s5
        // CHECK: d5 * s5
        // CHECK: d5 mod s5
        // CHECK: d5 floordiv s5
        // CHECK: d5 ceildiv s5

        // Tests methods of affine binary operation expression, takes add expression
        // as an example.
        mlirAffineExprDump(mlirAffineBinaryOpExprGetLHS(affineAddExpr));
        mlirAffineExprDump(mlirAffineBinaryOpExprGetRHS(affineAddExpr));
        // CHECK: d5
        // CHECK: s5

        // Tests methods of affine dimension expression.
        if mlirAffineDimExprGetPosition(affineDimExpr) != 5 {
            return 1;
        }

        // Tests methods of affine symbol expression.
        if mlirAffineSymbolExprGetPosition(affineSymbolExpr) != 5 {
            return 2;
        }

        // Tests methods of affine constant expression.
        if mlirAffineConstantExprGetValue(affineConstantExpr) != 5 {
            return 3;
        }

        // Tests methods of affine expression.
        if 0 != mlirAffineExprIsSymbolicOrConstant(affineDimExpr)
            || 0 == mlirAffineExprIsSymbolicOrConstant(affineSymbolExpr)
            || 0 == mlirAffineExprIsSymbolicOrConstant(affineConstantExpr)
            || 0 != mlirAffineExprIsSymbolicOrConstant(affineAddExpr)
            || 0 != mlirAffineExprIsSymbolicOrConstant(affineMulExpr)
            || 0 != mlirAffineExprIsSymbolicOrConstant(affineModExpr)
            || 0 != mlirAffineExprIsSymbolicOrConstant(affineFloorDivExpr)
            || 0 != mlirAffineExprIsSymbolicOrConstant(affineCeilDivExpr)
        {
            return 4;
        }

        if 0 == mlirAffineExprIsPureAffine(affineDimExpr)
            || 0 == mlirAffineExprIsPureAffine(affineSymbolExpr)
            || 0 == mlirAffineExprIsPureAffine(affineConstantExpr)
            || 0 == mlirAffineExprIsPureAffine(affineAddExpr)
            || 0 != mlirAffineExprIsPureAffine(affineMulExpr)
            || 0 != mlirAffineExprIsPureAffine(affineModExpr)
            || 0 != mlirAffineExprIsPureAffine(affineFloorDivExpr)
            || 0 != mlirAffineExprIsPureAffine(affineCeilDivExpr)
        {
            return 5;
        }

        if mlirAffineExprGetLargestKnownDivisor(affineDimExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineSymbolExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineConstantExpr) != 5
            || mlirAffineExprGetLargestKnownDivisor(affineAddExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineMulExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineModExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineFloorDivExpr) != 1
            || mlirAffineExprGetLargestKnownDivisor(affineCeilDivExpr) != 1
        {
            return 6;
        }

        if 0 == mlirAffineExprIsMultipleOf(affineDimExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineSymbolExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineConstantExpr, 5)
            || 0 == mlirAffineExprIsMultipleOf(affineAddExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineMulExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineModExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineFloorDivExpr, 1)
            || 0 == mlirAffineExprIsMultipleOf(affineCeilDivExpr, 1)
        {
            return 7;
        }

        if 0 == mlirAffineExprIsFunctionOfDim(affineDimExpr, 5)
            || 0 != mlirAffineExprIsFunctionOfDim(affineSymbolExpr, 5)
            || 0 != mlirAffineExprIsFunctionOfDim(affineConstantExpr, 5)
            || 0 == mlirAffineExprIsFunctionOfDim(affineAddExpr, 5)
            || 0 == mlirAffineExprIsFunctionOfDim(affineMulExpr, 5)
            || 0 == mlirAffineExprIsFunctionOfDim(affineModExpr, 5)
            || 0 == mlirAffineExprIsFunctionOfDim(affineFloorDivExpr, 5)
            || 0 == mlirAffineExprIsFunctionOfDim(affineCeilDivExpr, 5)
        {
            return 8;
        }

        // Tests 'IsA' methods of affine binary operation expression.
        if 0 == mlirAffineExprIsAAdd(affineAddExpr) {
            return 9;
        }

        if 0 == mlirAffineExprIsAMul(affineMulExpr) {
            return 10;
        }

        if 0 == mlirAffineExprIsAMod(affineModExpr) {
            return 11;
        }

        if 0 == mlirAffineExprIsAFloorDiv(affineFloorDivExpr) {
            return 12;
        }

        if 0 == mlirAffineExprIsACeilDiv(affineCeilDivExpr) {
            return 13;
        }

        if 0 == mlirAffineExprIsABinary(affineAddExpr) {
            return 14;
        }

        // Test other 'IsA' method on affine expressions.
        if 0 == mlirAffineExprIsAConstant(affineConstantExpr) {
            return 15;
        }

        if 0 == mlirAffineExprIsADim(affineDimExpr) {
            return 16;
        }

        if 0 == mlirAffineExprIsASymbol(affineSymbolExpr) {
            return 17;
        }

        // Test equality and nullity.
        let otherDimExpr = mlirAffineDimExprGet(ctx, 5);
        if 0 == mlirAffineExprEqual(affineDimExpr, otherDimExpr) {
            return 18;
        }

        if affineDimExpr.ptr == std::ptr::null() {
            return 19;
        }

        0
    }
}

fn affineMapFromExprs(ctx: MlirContext) -> i32 {
    unsafe {
        let affineDimExpr = mlirAffineDimExprGet(ctx, 0);
        let affineSymbolExpr = mlirAffineSymbolExprGet(ctx, 1);
        let mut exprs = [affineDimExpr, affineSymbolExpr];
        let map = mlirAffineMapGet(ctx, 3, 3, 2, exprs.as_mut_ptr());

        // CHECK-LABEL: @affineMapFromExprs
        eprint!("@affineMapFromExprs");
        // CHECK: (d0, d1, d2)[s0, s1, s2] -> (d0, s1)
        mlirAffineMapDump(map);

        if mlirAffineMapGetNumResults(map) != 2 {
            return 1;
        }

        if 0 == mlirAffineExprEqual(mlirAffineMapGetResult(map, 0), affineDimExpr) {
            return 2;
        }
        if 0 == mlirAffineExprEqual(mlirAffineMapGetResult(map, 1), affineSymbolExpr) {
            return 3;
        }

        let affineDim2Expr = mlirAffineDimExprGet(ctx, 1);
        let composed = mlirAffineExprCompose(affineDim2Expr, map);
        // CHECK: s1
        mlirAffineExprDump(composed);
        if 0 == mlirAffineExprEqual(composed, affineSymbolExpr) {
            return 4;
        }

        0
    }
}

fn printIntegerSet(ctx: MlirContext) -> i32 {
    unsafe {
        let emptySet = mlirIntegerSetEmptyGet(ctx, 2, 1);

        // CHECK-LABEL: @printIntegerSet
        eprint!("@printIntegerSet");

        // CHECK: (d0, d1)[s0] : (1 == 0)
        mlirIntegerSetDump(emptySet);

        if 0 == mlirIntegerSetIsCanonicalEmpty(emptySet) {
            return 1;
        }
        let anotherEmptySet = mlirIntegerSetEmptyGet(ctx, 2, 1);
        if 0 == mlirIntegerSetEqual(emptySet, anotherEmptySet) {
            return 2;
        }

        // Construct a set constrained by:
        //   d0 - s0 == 0,
        //   d1 - 42 >= 0.
        let negOne = mlirAffineConstantExprGet(ctx, -1);
        let negFortyTwo = mlirAffineConstantExprGet(ctx, -42);
        let d0 = mlirAffineDimExprGet(ctx, 0);
        let d1 = mlirAffineDimExprGet(ctx, 1);
        let s0 = mlirAffineSymbolExprGet(ctx, 0);
        let negS0 = mlirAffineMulExprGet(negOne, s0);
        let d0minusS0 = mlirAffineAddExprGet(d0, negS0);
        let d1minus42 = mlirAffineAddExprGet(d1, negFortyTwo);
        let constraints = [d0minusS0, d1minus42];
        let flags = [true as u8, false as u8];

        let set = mlirIntegerSetGet(ctx, 2, 1, 2, constraints.as_ptr(), flags.as_ptr());
        // CHECK: (d0, d1)[s0] : (
        // CHECK-DAG: d0 - s0 == 0
        // CHECK-DAG: d1 - 42 >= 0
        mlirIntegerSetDump(set);

        // Transform d1 into s0.
        let s1 = mlirAffineSymbolExprGet(ctx, 1);
        let repl = [d0, s1];
        let replaced = mlirIntegerSetReplaceGet(set, repl.as_ptr(), &s0, 1, 2);
        // CHECK: (d0)[s0, s1] : (
        // CHECK-DAG: d0 - s0 == 0
        // CHECK-DAG: s1 - 42 >= 0
        mlirIntegerSetDump(replaced);

        if mlirIntegerSetGetNumDims(set) != 2 {
            return 3;
        }
        if mlirIntegerSetGetNumDims(replaced) != 1 {
            return 4;
        }

        if mlirIntegerSetGetNumSymbols(set) != 1 {
            return 5;
        }
        if mlirIntegerSetGetNumSymbols(replaced) != 2 {
            return 6;
        }

        if mlirIntegerSetGetNumInputs(set) != 3 {
            return 7;
        }

        if mlirIntegerSetGetNumConstraints(set) != 2 {
            return 8;
        }

        if mlirIntegerSetGetNumEqualities(set) != 1 {
            return 9;
        }

        if mlirIntegerSetGetNumInequalities(set) != 1 {
            return 10;
        }

        let cstr1 = mlirIntegerSetGetConstraint(set, 0);
        let cstr2 = mlirIntegerSetGetConstraint(set, 1);
        let isEq1 = mlirIntegerSetIsConstraintEq(set, 0);
        let isEq2 = mlirIntegerSetIsConstraintEq(set, 1);
        if 0 == mlirAffineExprEqual(cstr1, if isEq1 != 0 { d0minusS0 } else { d1minus42 }) {
            return 11;
        }
        if 0 == mlirAffineExprEqual(cstr2, if isEq2 != 0 { d0minusS0 } else { d1minus42 }) {
            return 12;
        }

        0
    }
}

fn registerOnlyStd() -> i32 {
    unsafe {
        let ctx = mlirContextCreate();
        // The built-in dialect is always loaded.
        if mlirContextGetNumLoadedDialects(ctx) != 1 {
            return 1;
        }

        let stdHandle = mlirGetDialectHandle__func__();

        let mut stdDialect =
            mlirContextGetOrLoadDialect(ctx, mlirDialectHandleGetNamespace(stdHandle));
        if stdDialect.ptr != std::ptr::null_mut() {
            return 2;
        }

        mlirDialectHandleRegisterDialect(stdHandle, ctx);

        stdDialect = mlirContextGetOrLoadDialect(ctx, mlirDialectHandleGetNamespace(stdHandle));
        if stdDialect.ptr == std::ptr::null_mut() {
            return 3;
        }

        let alsoStd = mlirDialectHandleLoadDialect(stdHandle, ctx);
        if 0 == mlirDialectEqual(stdDialect, alsoStd) {
            return 4;
        }

        let stdNs = mlirDialectGetNamespace(stdDialect);
        let alsoStdNs = mlirDialectHandleGetNamespace(stdHandle);
        if stdNs.length != alsoStdNs.length
            || 0 != libc::strncmp(stdNs.data, alsoStdNs.data, stdNs.length as usize)
        {
            return 5;
        }

        eprintln!("@registration");
        // CHECK-LABEL: @registration

        // CHECK: func.call is_registered: 1
        eprintln!(
            "func.call is_registered: {}",
            mlirContextIsRegisteredOperation(
                ctx,
                mlirStringRefCreateFromCString("func.call\0".as_ptr() as *const i8)
            )
        );

        // CHECK: func.not_existing_op is_registered: 0
        eprintln!(
            "func.not_existing_op is_registered: {}",
            mlirContextIsRegisteredOperation(
                ctx,
                mlirStringRefCreateFromCString("func.not_existing_op\0".as_ptr() as *const i8)
            )
        );

        // CHECK: not_existing_dialect.not_existing_op is_registered: 0
        eprintln!(
            "not_existing_dialect.not_existing_op is_registered: {}",
            mlirContextIsRegisteredOperation(
                ctx,
                mlirStringRefCreateFromCString(
                    "not_existing_dialect.not_existing_op\0".as_ptr() as *const i8
                )
            )
        );

        mlirContextDestroy(ctx);
        0
    }
}

fn testBackreferences() -> i32 {
    unsafe {
        eprintln!("@test_backreferences");

        let ctx = mlirContextCreate();
        mlirContextSetAllowUnregisteredDialects(ctx, 1u8);
        let loc = mlirLocationUnknownGet(ctx);

        let mut opState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("invalid.op\0".as_ptr() as *const i8),
            loc,
        );
        let region = mlirRegionCreate();
        let block = mlirBlockCreate(0, std::ptr::null(), std::ptr::null());
        mlirRegionAppendOwnedBlock(region, block);
        mlirOperationStateAddOwnedRegions(&mut opState, 1, &region);
        let op = mlirOperationCreate(&mut opState);
        let ident = mlirIdentifierGet(
            ctx,
            mlirStringRefCreateFromCString("identifier\0".as_ptr() as *const i8),
        );

        if 0 == mlirContextEqual(ctx, mlirOperationGetContext(op)) {
            eprintln!("ERROR: Getting context from operation failed");
            return 1;
        }
        if 0 == mlirOperationEqual(op, mlirBlockGetParentOperation(block)) {
            eprintln!("ERROR: Getting parent operation from block failed");
            return 2;
        }
        if 0 == mlirContextEqual(ctx, mlirIdentifierGetContext(ident)) {
            eprintln!("ERROR: Getting context from identifier failed");
            return 3;
        }

        mlirOperationDestroy(op);
        mlirContextDestroy(ctx);

        // CHECK-LABEL: @test_backreferences
        0
    }
}

/// Tests operand APIs.
fn testOperands() -> i32 {
    unsafe {
        eprintln!("@testOperands");
        // CHECK-LABEL: @testOperands

        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("arith\0".as_ptr() as *const i8),
        );
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("test\0".as_ptr() as *const i8),
        );
        let loc = mlirLocationUnknownGet(ctx);
        let indexType = mlirIndexTypeGet(ctx);

        // Create some constants to use as operands.
        let indexZeroLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("0 : index\0".as_ptr() as *const i8),
        );
        let indexZeroValueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            indexZeroLiteral,
        );
        let mut constZeroState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddResults(&mut constZeroState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constZeroState, 1, &indexZeroValueAttr);
        let constZero = mlirOperationCreate(&mut constZeroState);
        let constZeroValue = mlirOperationGetResult(constZero, 0);

        let indexOneLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("1 : index\0".as_ptr() as *const i8),
        );
        let indexOneValueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            indexOneLiteral,
        );
        let mut constOneState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddResults(&mut constOneState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constOneState, 1, &indexOneValueAttr);
        let constOne = mlirOperationCreate(&mut constOneState);
        let constOneValue = mlirOperationGetResult(constOne, 0);

        // Create the operation under test.
        mlirContextSetAllowUnregisteredDialects(ctx, 1u8);
        let mut opState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op\0".as_ptr() as *const i8),
            loc,
        );
        let initialOperands = [constZeroValue];
        mlirOperationStateAddOperands(&mut opState, 1, initialOperands.as_ptr());
        let op = mlirOperationCreate(&mut opState);

        // Test operand APIs.
        let numOperands = mlirOperationGetNumOperands(op);
        eprintln!("Num Operands: {}", numOperands);
        // CHECK: Num Operands: 1

        let opOperand1 = mlirOperationGetOperand(op, 0);
        eprint!("Original operand: ");
        mlirValuePrint(
            opOperand1,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        // CHECK: Original operand: {{.+}} arith.constant 0 : index

        mlirOperationSetOperand(op, 0, constOneValue);
        let opOperand2 = mlirOperationGetOperand(op, 0);
        eprint!("Updated operand: ");
        mlirValuePrint(
            opOperand2,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        // CHECK: Updated operand: {{.+}} arith.constant 1 : index

        // Test op operand APIs.
        let use1 = mlirValueGetFirstUse(opOperand1);
        if 0 == mlirOpOperandIsNull(use1) {
            eprintln!("ERROR: Use should be null");
            return 1;
        }

        let mut use2 = mlirValueGetFirstUse(opOperand2);
        if 0 != mlirOpOperandIsNull(use2) {
            eprintln!("ERROR: Use should not be null");
            return 2;
        }

        eprint!("Use owner: ");
        mlirOperationPrint(
            mlirOpOperandGetOwner(use2),
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Use owner: "dummy.op"

        eprintln!("Use operandNumber: {}", mlirOpOperandGetOperandNumber(use2));
        // CHECK: Use operandNumber: 0

        use2 = mlirOpOperandGetNextUse(use2);
        if 0 == mlirOpOperandIsNull(use2) {
            eprintln!("ERROR: Next use should be null");
            return 3;
        }

        let mut op2State = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op2\0".as_ptr() as *const i8),
            loc,
        );
        let initialOperands2 = [constOneValue];
        mlirOperationStateAddOperands(&mut op2State, 1, initialOperands2.as_ptr());
        let op2 = mlirOperationCreate(&mut op2State);

        let mut use3 = mlirValueGetFirstUse(constOneValue);
        eprint!("First use owner: ");
        mlirOperationPrint(
            mlirOpOperandGetOwner(use3),
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: First use owner: "dummy.op2"

        use3 = mlirOpOperandGetNextUse(mlirValueGetFirstUse(constOneValue));
        eprint!("Second use owner: ");
        mlirOperationPrint(
            mlirOpOperandGetOwner(use3),
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Second use owner: "dummy.op"

        let indexTwoLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("2 : index\0".as_ptr() as *const i8),
        );
        let indexTwoValueAttr = mlirNamedAttributeGet(
            mlirIdentifierGet(
                ctx,
                mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8),
            ),
            indexTwoLiteral,
        );
        let mut constTwoState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddResults(&mut constTwoState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constTwoState, 1, &indexTwoValueAttr);
        let constTwo = mlirOperationCreate(&mut constTwoState);
        let constTwoValue = mlirOperationGetResult(constTwo, 0);

        mlirValueReplaceAllUsesOfWith(constOneValue, constTwoValue);

        use3 = mlirValueGetFirstUse(constOneValue);
        if 0 == mlirOpOperandIsNull(use3) {
            eprintln!("ERROR: Use should be null");
            return 4;
        }

        let mut use4 = mlirValueGetFirstUse(constTwoValue);
        eprint!("First replacement use owner: ");
        mlirOperationPrint(
            mlirOpOperandGetOwner(use4),
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: First replacement use owner: "dummy.op"

        use4 = mlirOpOperandGetNextUse(mlirValueGetFirstUse(constTwoValue));
        eprint!("Second replacement use owner: ");
        mlirOperationPrint(
            mlirOpOperandGetOwner(use4),
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        eprintln!();
        // CHECK: Second replacement use owner: "dummy.op2"

        let use5 = mlirValueGetFirstUse(constTwoValue);
        let use6 = mlirOpOperandGetNextUse(use5);
        if 0 == mlirValueEqual(mlirOpOperandGetValue(use5), mlirOpOperandGetValue(use6)) {
            eprintln!("ERROR: First and second operand should share the same value");
            return 5;
        }

        mlirOperationDestroy(op);
        mlirOperationDestroy(op2);
        mlirOperationDestroy(constZero);
        mlirOperationDestroy(constOne);
        mlirOperationDestroy(constTwo);
        mlirContextDestroy(ctx);

        0
    }
}

/// Tests clone APIs.
fn testClone() -> i32 {
    unsafe {
        eprintln!("@testClone");
        // CHECK-LABEL: @testClone

        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);

        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("func\0".as_ptr() as *const i8),
        );
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("arith\0".as_ptr() as *const i8),
        );
        let loc = mlirLocationUnknownGet(ctx);
        let indexType = mlirIndexTypeGet(ctx);
        let valueStringRef = mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8);

        let indexZeroLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("0 : index\0".as_ptr() as *const i8),
        );
        let indexZeroValueAttr =
            mlirNamedAttributeGet(mlirIdentifierGet(ctx, valueStringRef), indexZeroLiteral);
        let mut constZeroState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddResults(&mut constZeroState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constZeroState, 1, &indexZeroValueAttr);
        let constZero = mlirOperationCreate(&mut constZeroState);

        let indexOneLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("1 : index\0".as_ptr() as *const i8),
        );
        let constOne = mlirOperationClone(constZero);
        mlirOperationSetAttributeByName(constOne, valueStringRef, indexOneLiteral);

        mlirOperationPrint(
            constZero,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        mlirOperationPrint(
            constOne,
            printToStderr as MlirStringCallback,
            std::ptr::null_mut(),
        );
        // CHECK: arith.constant 0 : index
        // CHECK: arith.constant 1 : index

        mlirOperationDestroy(constZero);
        mlirOperationDestroy(constOne);
        mlirContextDestroy(ctx);
        0
    }
}

fn testTypeID(ctx: MlirContext) -> i32 {
    unsafe {
        eprintln!("@testTypeID");

        // Test getting and comparing type and attribute type ids.
        let i32Ty = mlirIntegerTypeGet(ctx, 32);
        let i32ID = mlirTypeGetTypeID(i32Ty);
        let ui32 = mlirIntegerTypeUnsignedGet(ctx, 32);
        let ui32ID = mlirTypeGetTypeID(ui32);
        let f32Ty = mlirF32TypeGet(ctx);
        let f32ID = mlirTypeGetTypeID(f32Ty);
        let i32Attr = mlirIntegerAttrGet(i32Ty, 1);
        let i32AttrID = mlirAttributeGetTypeID(i32Attr);

        if i32ID.ptr == std::ptr::null_mut()
            || ui32ID.ptr == std::ptr::null_mut()
            || f32ID.ptr == std::ptr::null_mut()
            || i32AttrID.ptr == std::ptr::null_mut()
        {
            eprintln!("ERROR: Expected type ids to be present");
            return 1;
        }

        if 0 == mlirTypeIDEqual(i32ID, ui32ID)
            || mlirTypeIDHashValue(i32ID) != mlirTypeIDHashValue(ui32ID)
        {
            eprintln!("ERROR: Expected different integer types to have the same type id");
            return 2;
        }

        if 0 != mlirTypeIDEqual(i32ID, f32ID) {
            eprintln!("ERROR: Expected integer type id to not equal float type id");
            return 3;
        }

        if 0 != mlirTypeIDEqual(i32ID, i32AttrID) {
            eprintln!("ERROR: Expected integer type id to not equal integer attribute type id");
            return 4;
        }

        let loc = mlirLocationUnknownGet(ctx);
        let indexType = mlirIndexTypeGet(ctx);
        let valueStringRef = mlirStringRefCreateFromCString("value\0".as_ptr() as *const i8);

        // Create a registered operation, which should have a type id.
        let indexZeroLiteral = mlirAttributeParseGet(
            ctx,
            mlirStringRefCreateFromCString("0 : index\0".as_ptr() as *const i8),
        );
        let indexZeroValueAttr =
            mlirNamedAttributeGet(mlirIdentifierGet(ctx, valueStringRef), indexZeroLiteral);
        let mut constZeroState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("arith.constant\0".as_ptr() as *const i8),
            loc,
        );
        mlirOperationStateAddResults(&mut constZeroState, 1, &indexType);
        mlirOperationStateAddAttributes(&mut constZeroState, 1, &indexZeroValueAttr);
        let constZero = mlirOperationCreate(&mut constZeroState);

        if 0 == mlirOperationVerify(constZero) {
            eprintln!("ERROR: Expected operation to verify correctly");
            return 5;
        }

        if constZero.ptr == std::ptr::null_mut() {
            eprintln!("ERROR: Expected registered operation to be present");
            return 6;
        }

        let registeredOpID = mlirOperationGetTypeID(constZero);

        if registeredOpID.ptr == std::ptr::null_mut() {
            eprintln!("ERROR: Expected registered operation type id to be present");
            return 7;
        }

        // Create an unregistered operation, which should not have a type id.
        mlirContextSetAllowUnregisteredDialects(ctx, 1u8);
        let mut opState = mlirOperationStateGet(
            mlirStringRefCreateFromCString("dummy.op\0".as_ptr() as *const i8),
            loc,
        );
        let unregisteredOp = mlirOperationCreate(&mut opState);
        if unregisteredOp.ptr == std::ptr::null_mut() {
            eprintln!("ERROR: Expected unregistered operation to be present");
            return 8;
        }

        let unregisteredOpID = mlirOperationGetTypeID(unregisteredOp);

        if unregisteredOpID.ptr != std::ptr::null_mut() {
            eprintln!("ERROR: Expected unregistered operation type id to be null");
            return 9;
        }

        mlirOperationDestroy(constZero);
        mlirOperationDestroy(unregisteredOp);

        0
    }
}

fn mlirOperationIsNull(op: MlirOperation) -> bool {
    op.ptr == std::ptr::null_mut()
}
fn testSymbolTable(ctx: MlirContext) -> i32 {
    unsafe {
        eprintln!("@testSymbolTable");

        let moduleString =
            "func.func private @foo() func.func private @bar()\0".as_ptr() as *const i8;
        let otherModuleString =
            "func.func private @qux() func.func private @foo()\0".as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let otherModule =
            mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(otherModuleString));

        let symbolTable = mlirSymbolTableCreate(mlirModuleGetOperation(module));

        let funcFoo = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        );
        if mlirOperationIsNull(funcFoo) {
            return 1;
        }
        let funcBar = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
        );
        if 0 != mlirOperationEqual(funcFoo, funcBar) {
            return 2;
        }

        let missing = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("qux\0".as_ptr() as *const i8),
        );
        if !mlirOperationIsNull(missing) {
            return 3;
        }

        let moduleBody = mlirModuleGetBody(module);
        let otherModuleBody = mlirModuleGetBody(otherModule);
        let operation = mlirBlockGetFirstOperation(otherModuleBody);
        mlirOperationRemoveFromParent(operation);
        mlirBlockAppendOwnedOperation(moduleBody, operation);

        // At this moment, the operation is still missing from the symbol table.
        let stillMissing = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("qux\0".as_ptr() as *const i8),
        );
        if !mlirOperationIsNull(stillMissing) {
            return 4;
        }

        // After it is added to the symbol table, and not only the operation with
        // which the table is associated, it can be looked up.
        mlirSymbolTableInsert(symbolTable, operation);
        let funcQux = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("qux\0".as_ptr() as *const i8),
        );
        if 0 == mlirOperationEqual(operation, funcQux) {
            return 5;
        }

        // Erasing from the symbol table also removes the operation.
        mlirSymbolTableErase(symbolTable, funcBar);
        let nowMissing = mlirSymbolTableLookup(
            symbolTable,
            mlirStringRefCreateFromCString("bar\0".as_ptr() as *const i8),
        );
        if !mlirOperationIsNull(nowMissing) {
            return 6;
        }

        // Adding a symbol with the same name to the table should rename.
        let duplicateNameOp = mlirBlockGetFirstOperation(otherModuleBody);
        mlirOperationRemoveFromParent(duplicateNameOp);
        mlirBlockAppendOwnedOperation(moduleBody, duplicateNameOp);
        let newName = mlirSymbolTableInsert(symbolTable, duplicateNameOp);
        let newNameStr = mlirStringAttrGetValue(newName);
        if 0 != mlirStringRefEqual(
            newNameStr,
            mlirStringRefCreateFromCString("foo\0".as_ptr() as *const i8),
        ) {
            return 7;
        }
        let updatedName = mlirOperationGetAttributeByName(
            duplicateNameOp,
            mlirSymbolTableGetSymbolAttributeName(),
        );
        if 0 == mlirAttributeEqual(updatedName, newName) {
            return 8;
        }

        mlirOperationDump(mlirModuleGetOperation(module));
        mlirOperationDump(mlirModuleGetOperation(otherModule));
        // clang-format off
        // CHECK-LABEL: @testSymbolTable
        // CHECK: module
        // CHECK:   func private @foo
        // CHECK:   func private @qux
        // CHECK:   func private @foo{{.+}}
        // CHECK: module
        // CHECK-NOT: @qux
        // CHECK-NOT: @foo
        // clang-format on

        mlirSymbolTableDestroy(symbolTable);
        mlirModuleDestroy(module);
        mlirModuleDestroy(otherModule);

        0
    }
}

fn mlirDialectRegistryIsNull(registry: MlirDialectRegistry) -> bool {
    registry.ptr == std::ptr::null_mut()
}

fn testDialectRegistry() -> i32 {
    unsafe {
        eprintln!("@testDialectRegistry");

        let registry = mlirDialectRegistryCreate();
        if mlirDialectRegistryIsNull(registry) {
            eprintln!("ERROR: Expected registry to be present");
            return 1;
        }

        let stdHandle = mlirGetDialectHandle__func__();
        mlirDialectHandleInsertDialect(stdHandle, registry);

        let ctx = mlirContextCreate();
        if mlirContextGetNumRegisteredDialects(ctx) != 0 {
            eprintln!("ERROR: Expected no dialects to be registered to new context");
        }

        mlirContextAppendDialectRegistry(ctx, registry);
        if mlirContextGetNumRegisteredDialects(ctx) != 1 {
            eprintln!(
                "ERROR: Expected the dialect in the registry to be registered to the context"
            );
        }

        mlirContextDestroy(ctx);
        mlirDialectRegistryDestroy(registry);

        0
    }
}

#[allow(non_camel_case_types)]
struct callBackData {
    pub x: *const i8,
}

fn walkCallBack(op: MlirOperation, rootOpVoid: *const u8) -> MlirWalkResult {
    unsafe {
        let x = (*(rootOpVoid as *const callBackData)).x;
        let l = libc::strlen(x);
        let x_slice = std::slice::from_raw_parts(x as *const u8, l);
        let op_data = mlirIdentifierStr(mlirOperationGetName(op));
        let op_data_slice =
            std::slice::from_raw_parts(op_data.data as *const u8, op_data.length as usize);
        let x_str = std::str::from_utf8_unchecked(x_slice);
        let op_data_str = std::str::from_utf8_unchecked(op_data_slice);
        eprintln!("{}: {}", x_str, op_data_str);
        MlirWalkResultAdvance
    }
}

fn walkCallBackTestWalkResult(op: MlirOperation, rootOpVoid: *const u8) -> MlirWalkResult {
    unsafe {
        let x = (*(rootOpVoid as *const callBackData)).x;
        let l = libc::strlen(x);
        let x_slice = std::slice::from_raw_parts(x as *const u8, l);
        let op_data = mlirIdentifierStr(mlirOperationGetName(op));
        let op_data_slice =
            std::slice::from_raw_parts(op_data.data as *const u8, op_data.length as usize);
        let x_str = std::str::from_utf8_unchecked(x_slice);
        let op_data_str = std::str::from_utf8_unchecked(op_data_slice);
        eprintln!("{}: {}", x_str, op_data_str,);
        if libc::strcmp(
            mlirIdentifierStr(mlirOperationGetName(op)).data,
            "func.func\0".as_ptr() as *const i8,
        ) == 0
        {
            return MlirWalkResultSkip;
        }
        if libc::strcmp(
            mlirIdentifierStr(mlirOperationGetName(op)).data,
            "arith.addi\0".as_ptr() as *const i8,
        ) == 0
        {
            return MlirWalkResultInterrupt;
        }
        MlirWalkResultAdvance
    }
}

fn testOperationWalk(ctx: MlirContext) -> i32 {
    unsafe {
        // CHECK-LABEL: @testOperationWalk
        eprintln!("@testOperationWalk");

        let moduleString = "module {
func.func @foo() {
    %1 = arith.constant 10: i32
    arith.addi %1, %1: i32
    return
  }
  func.func @bar() {
    return
  }
}\0"
        .as_ptr() as *const i8;
        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));

        let mut data = callBackData {
            x: std::ptr::null(),
        };
        data.x = "i love you\0".as_ptr() as *const i8;

        // CHECK-NEXT: i love you: arith.constant
        // CHECK-NEXT: i love you: arith.addi
        // CHECK-NEXT: i love you: func.return
        // CHECK-NEXT: i love you: func.func
        // CHECK-NEXT: i love you: func.return
        // CHECK-NEXT: i love you: func.func
        // CHECK-NEXT: i love you: builtin.module
        mlirOperationWalk(
            mlirModuleGetOperation(module),
            walkCallBack as *mut extern "C" fn(StructMlirOperation, *mut std::ffi::c_void) -> u32,
            &mut data as *mut _ as *mut std::ffi::c_void,
            MlirWalkPostOrder,
        );

        data.x = "i don't love you\0".as_ptr() as *const i8;
        // CHECK-NEXT: i don't love you: builtin.module
        // CHECK-NEXT: i don't love you: func.func
        // CHECK-NEXT: i don't love you: arith.constant
        // CHECK-NEXT: i don't love you: arith.addi
        // CHECK-NEXT: i don't love you: func.return
        // CHECK-NEXT: i don't love you: func.func
        // CHECK-NEXT: i don't love you: func.return
        mlirOperationWalk(
            mlirModuleGetOperation(module),
            walkCallBack as *mut extern "C" fn(StructMlirOperation, *mut std::ffi::c_void) -> u32,
            &mut data as *mut _ as *mut std::ffi::c_void,
            MlirWalkPreOrder,
        );

        data.x = "interrupt\0".as_ptr() as *const i8;
        // Interrupted at `arith.addi`
        // CHECK-NEXT: interrupt: arith.constant
        // CHECK-NEXT: interrupt: arith.addi
        mlirOperationWalk(
            mlirModuleGetOperation(module),
            walkCallBackTestWalkResult
                as *mut extern "C" fn(StructMlirOperation, *mut std::ffi::c_void) -> u32,
            &mut data as *mut _ as *mut std::ffi::c_void,
            MlirWalkPostOrder,
        );

        data.x = "skip\0".as_ptr() as *const i8;
        // Skip at `func.func`
        // CHECK-NEXT: skip: builtin.module
        // CHECK-NEXT: skip: func.func
        // CHECK-NEXT: skip: func.func
        mlirOperationWalk(
            mlirModuleGetOperation(module),
            walkCallBackTestWalkResult
                as *mut extern "C" fn(StructMlirOperation, *mut std::ffi::c_void) -> u32,
            &mut data as *mut _ as *mut std::ffi::c_void,
            MlirWalkPreOrder,
        );

        mlirModuleDestroy(module);
        0
    }
}

fn testExplicitThreadPools() {
    unsafe {
        let threadPool = mlirLlvmThreadPoolCreate();
        let registry = mlirDialectRegistryCreate();
        mlirRegisterAllDialects(registry);
        let context = mlirContextCreateWithRegistry(registry, /*threadingEnabled=*/ 0);
        mlirContextSetThreadPool(context, threadPool);
        mlirContextDestroy(context);
        mlirDialectRegistryDestroy(registry);
        mlirLlvmThreadPoolDestroy(threadPool);
    }
}

fn mlirLogicalResultSuccess() -> MlirLogicalResult {
    MlirLogicalResult { value: 1 }
}
// Wraps a diagnostic into additional text we can match against.
pub extern "C" fn errorHandler(
    diagnostic: MlirDiagnostic,
    userDataPtr: *const u8,
) -> MlirLogicalResult {
    unsafe {
        let userData = userDataPtr as i64;
        eprintln!("processing diagnostic (userData: {}) <<", userData);
        mlirDiagnosticPrint(diagnostic, printToStderr as _, std::ptr::null_mut());
        eprintln!();
        let loc = mlirDiagnosticGetLocation(diagnostic);
        mlirLocationPrint(loc, printToStderr as _, std::ptr::null_mut());
        assert!(mlirDiagnosticGetNumNotes(diagnostic) == 0);
        eprint!("\n>> end of diagnostic (userData: {})\n", userData);
        mlirLogicalResultSuccess()
    }
}

// Logs when the delete user data callback is called
pub extern "C" fn deleteUserData(userData: *const u8) {
    eprintln!("deleting user data (userData: {})", userData as u64);
}

fn testDiagnostics() {
    unsafe {
        let ctx = mlirContextCreate();
        let id = mlirContextAttachDiagnosticHandler(
            ctx,
            errorHandler as _,
            42_i64 as *mut std::ffi::c_void,
            deleteUserData as _,
        );
        eprintln!("@test_diagnostics");
        let unknownLoc = mlirLocationUnknownGet(ctx);
        mlirEmitError(unknownLoc, "test diagnostics\0".as_ptr() as *const i8);
        let unknownAttr = mlirLocationGetAttribute(unknownLoc);
        let unknownClone = mlirLocationFromAttribute(unknownAttr);
        mlirEmitError(unknownClone, "test clone\0".as_ptr() as *const i8);
        let fileLineColLoc = mlirLocationFileLineColGet(
            ctx,
            mlirStringRefCreateFromCString("file.c\0".as_ptr() as *const i8),
            1,
            2,
        );
        mlirEmitError(fileLineColLoc, "test diagnostics\0".as_ptr() as *const i8);
        let callSiteLoc = mlirLocationCallSiteGet(
            mlirLocationFileLineColGet(
                ctx,
                mlirStringRefCreateFromCString("other-file.c\0".as_ptr() as *const i8),
                2,
                3,
            ),
            fileLineColLoc,
        );
        mlirEmitError(callSiteLoc, "test diagnostics\0".as_ptr() as *const i8);
        let null = MlirLocation {
            ptr: std::ptr::null(),
        };
        let nameLoc = mlirLocationNameGet(
            ctx,
            mlirStringRefCreateFromCString("named\0".as_ptr() as *const i8),
            null,
        );
        mlirEmitError(nameLoc, "test diagnostics\0".as_ptr() as *const i8);
        let locs = [nameLoc, callSiteLoc];
        let nullAttr = MlirAttribute {
            ptr: std::ptr::null(),
        };
        let fusedLoc = mlirLocationFusedGet(ctx, 2, locs.as_ptr(), nullAttr);
        mlirEmitError(fusedLoc, "test diagnostics\0".as_ptr() as *const i8);
        mlirContextDetachDiagnosticHandler(ctx, id);
        mlirEmitError(unknownLoc, "more test diagnostics\0".as_ptr() as *const i8);
        // CHECK-LABEL: @test_diagnostics
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test diagnostics
        // CHECK:   loc(unknown)
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test clone
        // CHECK:   loc(unknown)
        // CHECK: >> end of diagnostic (userData: 42)
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test diagnostics
        // CHECK:   loc("file.c":1:2)
        // CHECK: >> end of diagnostic (userData: 42)
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test diagnostics
        // CHECK:   loc(callsite("other-file.c":2:3 at "file.c":1:2))
        // CHECK: >> end of diagnostic (userData: 42)
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test diagnostics
        // CHECK:   loc("named")
        // CHECK: >> end of diagnostic (userData: 42)
        // CHECK: processing diagnostic (userData: 42) <<
        // CHECK:   test diagnostics
        // CHECK:   loc(fused["named", callsite("other-file.c":2:3 at "file.c":1:2)])
        // CHECK: deleting user data (userData: 42)
        // CHECK-NOT: processing diagnostic
        // CHECK:     more test diagnostics
        mlirContextDestroy(ctx);
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        registerAllUpstreamDialects(ctx);
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("func\0".as_ptr() as *const i8),
        );
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("memref\0".as_ptr() as *const i8),
        );
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("shape\0".as_ptr() as *const i8),
        );
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("scf\0".as_ptr() as *const i8),
        );

        constructAndTraverseIr(ctx);
        buildWithInsertionsAndPrint(ctx);
        if 0 != createOperationWithTypeInference(ctx) {
            std::process::exit(2);
        }
        if 0 != printBuiltinTypes(ctx) {
            std::process::exit(3);
        }
        if 0 != printBuiltinAttributes(ctx) {
            std::process::exit(4);
        }
        if 0 != printAffineMap(ctx) {
            std::process::exit(5);
        }
        if 0 != printAffineExpr(ctx) {
            std::process::exit(6);
        }
        if 0 != affineMapFromExprs(ctx) {
            std::process::exit(7);
        }
        if 0 != printIntegerSet(ctx) {
            std::process::exit(8);
        }
        if 0 != registerOnlyStd() {
            std::process::exit(9);
        }
        if 0 != testBackreferences() {
            std::process::exit(10);
        }
        if 0 != testOperands() {
            std::process::exit(11);
        }
        if 0 != testClone() {
            std::process::exit(12);
        }
        if 0 != testTypeID(ctx) {
            std::process::exit(13);
        }
        if 0 != testSymbolTable(ctx) {
            std::process::exit(14);
        }
        if 0 != testDialectRegistry() {
            std::process::exit(15);
        }
        if 0 != testOperationWalk(ctx) {
            std::process::exit(16);
        }
        testExplicitThreadPools();
        testDiagnostics();
        // CHECK: DESTROY MAIN CONTEXT
        // CHECK: reportResourceDelete: resource_i64_blob
        eprintln!("DESTROY MAIN CONTEXT");
        mlirContextDestroy(ctx);
    }
}
