// RUN: bash %S/run_test.sh %s 2>&1 |%FileCheck %s
#![allow(non_snake_case)]

use mlir_capi::BuiltinTypes::*;
use mlir_capi::Rewrite::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

fn createOperationWithName(ctx: MlirContext, name: *const i8) -> MlirOperation {
    unsafe {
        let nameRef = mlirStringRefCreateFromCString(name);
        let loc = mlirLocationUnknownGet(ctx);
        let mut state = mlirOperationStateGet(nameRef, loc);
        let indexType = mlirIndexTypeGet(ctx);
        mlirOperationStateAddResults(&mut state, 1, &indexType);
        mlirOperationCreate(&mut state)
    }
}

fn testInsertionPoint(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testInsertionPoint
        eprintln!("@testInsertionPoint");

        let moduleString = "\"dialect.op1\"() : () -> ()\n\0".as_ptr() as *const i8;
        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);
        let op1 = mlirBlockGetFirstOperation(body);

        // IRRewriter create
        let rewriter = mlirIRRewriterCreate(ctx);

        // Insert before op
        mlirRewriterBaseSetInsertionPointBefore(rewriter, op1);
        let op2 = createOperationWithName(ctx, "dialect.op2\0".as_ptr() as *const i8);
        mlirRewriterBaseInsert(rewriter, op2);

        // Insert after op
        mlirRewriterBaseSetInsertionPointAfter(rewriter, op2);
        let op3 = createOperationWithName(ctx, "dialect.op3\0".as_ptr() as *const i8);
        mlirRewriterBaseInsert(rewriter, op3);
        let op3Res = mlirOperationGetResult(op3, 0);

        // Insert after value
        mlirRewriterBaseSetInsertionPointAfterValue(rewriter, op3Res);
        let op4 = createOperationWithName(ctx, "dialect.op4\0".as_ptr() as *const i8);
        mlirRewriterBaseInsert(rewriter, op4);

        // Insert at beginning of block
        mlirRewriterBaseSetInsertionPointToStart(rewriter, body);
        let op5 = createOperationWithName(ctx, "dialect.op5\0".as_ptr() as *const i8);
        mlirRewriterBaseInsert(rewriter, op5);

        // Insert at end of block
        mlirRewriterBaseSetInsertionPointToEnd(rewriter, body);
        let op6 = createOperationWithName(ctx, "dialect.op6\0".as_ptr() as *const i8);
        mlirRewriterBaseInsert(rewriter, op6);

        // Get insertion blocks
        let block1 = mlirRewriterBaseGetBlock(rewriter);
        let block2 = mlirRewriterBaseGetInsertionBlock(rewriter);
        assert!(body.ptr == block1.ptr);
        assert!(body.ptr == block2.ptr);

        // clang-format off
        // CHECK-NEXT: module {
        // CHECK-NEXT:   %{{.*}} = "dialect.op5"() : () -> index
        // CHECK-NEXT:   %{{.*}} = "dialect.op2"() : () -> index
        // CHECK-NEXT:   %{{.*}} = "dialect.op3"() : () -> index
        // CHECK-NEXT:   %{{.*}} = "dialect.op4"() : () -> index
        // CHECK-NEXT:   "dialect.op1"() : () -> ()
        // CHECK-NEXT:   %{{.*}} = "dialect.op6"() : () -> index
        // CHECK-NEXT: }
        // clang-format on
        mlirOperationDump(op);

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testCreateBlock(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testCreateBlock
        eprintln!("@testCreateBlock");

        let moduleString =
            "\"dialect.op1\"() ({^bb0:}) : () -> ()\n\"dialect.op2\"() ({^bb0:}) : () -> ()\n\0"
                .as_ptr() as *const i8;
        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        let op1 = mlirBlockGetFirstOperation(body);
        let region1 = mlirOperationGetRegion(op1, 0);
        let block1 = mlirRegionGetFirstBlock(region1);

        let op2 = mlirOperationGetNextInBlock(op1);
        let region2 = mlirOperationGetRegion(op2, 0);
        let block2 = mlirRegionGetFirstBlock(region2);

        let rewriter = mlirIRRewriterCreate(ctx);

        // Create block before
        let indexType = mlirIndexTypeGet(ctx);
        let unknown = mlirLocationUnknownGet(ctx);
        mlirRewriterBaseCreateBlockBefore(rewriter, block1, 1, &indexType, &unknown);

        mlirRewriterBaseSetInsertionPointToEnd(rewriter, body);

        // Clone operation
        mlirRewriterBaseClone(rewriter, op1);

        // Clone without regions
        mlirRewriterBaseCloneWithoutRegions(rewriter, op1);

        // Clone region before
        mlirRewriterBaseCloneRegionBefore(rewriter, region1, block2);

        mlirOperationDump(op);
        // clang-format off
        // CHECK-NEXT: "builtin.module"() ({
        // CHECK-NEXT:   "dialect.op1"() ({
        // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
        // CHECK-NEXT:   ^{{.*}}:
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op2"() ({
        // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
        // CHECK-NEXT:   ^{{.*}}:
        // CHECK-NEXT:   ^{{.*}}:
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op1"() ({
        // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
        // CHECK-NEXT:   ^{{.*}}:
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op1"() ({
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT: }) : () -> ()
        // clang-format on

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testInlineRegionBlock(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testInlineRegionBlock
        eprintln!("@testInlineRegionBlock");

        let moduleString = "\"dialect.op1\"() ({
  ^bb0(%arg0: index):
    \"dialect.op1_in1\"(%arg0) [^bb1] : (index) -> ()
  ^bb1():
    \"dialect.op1_in2\"() : () -> ()
}) : () -> ()
\"dialect.op2\"() ({^bb0:}) : () -> ()
\"dialect.op3\"() ({
  ^bb0(%arg0: index):
    \"dialect.op3_in1\"(%arg0) : (index) -> ()
  ^bb1():
    %x = \"dialect.op3_in2\"() : () -> index
    %y = \"dialect.op3_in3\"() : () -> index
}) : () -> ()
\"dialect.op4\"() ({
  ^bb0():
    \"dialect.op4_in1\"() : () -> index
  ^bb1(%arg0: index):
    \"dialect.op4_in2\"(%arg0) : (index) -> ()
}) : () -> ()\0"
            .as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        let op1 = mlirBlockGetFirstOperation(body);
        let region1 = mlirOperationGetRegion(op1, 0);

        let op2 = mlirOperationGetNextInBlock(op1);
        let region2 = mlirOperationGetRegion(op2, 0);
        let block2 = mlirRegionGetFirstBlock(region2);

        let op3 = mlirOperationGetNextInBlock(op2);
        let region3 = mlirOperationGetRegion(op3, 0);
        let block3_1 = mlirRegionGetFirstBlock(region3);
        let block3_2 = mlirBlockGetNextInRegion(block3_1);
        let op3_in2 = mlirBlockGetFirstOperation(block3_2);
        let op3_in2_res = mlirOperationGetResult(op3_in2, 0);
        let op3_in3 = mlirOperationGetNextInBlock(op3_in2);

        let op4 = mlirOperationGetNextInBlock(op3);
        let region4 = mlirOperationGetRegion(op4, 0);
        let block4_1 = mlirRegionGetFirstBlock(region4);
        let op4_in1 = mlirBlockGetFirstOperation(block4_1);
        let op4_in1_res = mlirOperationGetResult(op4_in1, 0);
        let block4_2 = mlirBlockGetNextInRegion(block4_1);

        let rewriter = mlirIRRewriterCreate(ctx);

        // Test these three functions
        mlirRewriterBaseInlineRegionBefore(rewriter, region1, block2);
        mlirRewriterBaseInlineBlockBefore(rewriter, block3_1, op3_in3, 1, &op3_in2_res);
        mlirRewriterBaseMergeBlocks(rewriter, block4_2, block4_1, 1, &op4_in1_res);

        mlirOperationDump(op);
        // clang-format off
        // CHECK-NEXT: "builtin.module"() ({
        // CHECK-NEXT:   "dialect.op1"() ({
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op2"() ({
        // CHECK-NEXT:   ^{{.*}}(%{{.*}}: index):
        // CHECK-NEXT:     "dialect.op1_in1"(%{{.*}})[^[[bb:.*]]] : (index) -> ()
        // CHECK-NEXT:   ^[[bb]]:
        // CHECK-NEXT:     "dialect.op1_in2"() : () -> ()
        // CHECK-NEXT:   ^{{.*}}:  // no predecessors
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op3"() ({
        // CHECK-NEXT:     %{{.*}} = "dialect.op3_in2"() : () -> index
        // CHECK-NEXT:     "dialect.op3_in1"(%{{.*}}) : (index) -> ()
        // CHECK-NEXT:     %{{.*}} = "dialect.op3_in3"() : () -> index
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op4"() ({
        // CHECK-NEXT:     %{{.*}} = "dialect.op4_in1"() : () -> index
        // CHECK-NEXT:     "dialect.op4_in2"(%{{.*}}) : (index) -> ()
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT: }) : () -> ()
        // clang-format on

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testReplaceOp(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testReplaceOp
        eprintln!("@testReplaceOp");

        let moduleString = "%x, %y, %z = \"dialect.create_values\"() : () -> (index, index, index)
%x_1, %y_1 = \"dialect.op1\"() : () -> (index, index)
\"dialect.use_op1\"(%x_1, %y_1) : (index, index) -> ()
%x_2, %y_2 = \"dialect.op2\"() : () -> (index, index)
%x_3, %y_3 = \"dialect.op3\"() : () -> (index, index)
\"dialect.use_op2\"(%x_2, %y_2) : (index, index) -> ()\n\0"
            .as_ptr() as *const i8;
        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        // get a handle to all operations/values
        let createValues = mlirBlockGetFirstOperation(body);
        let x = mlirOperationGetResult(createValues, 0);
        let z = mlirOperationGetResult(createValues, 2);
        let op1 = mlirOperationGetNextInBlock(createValues);
        let useOp1 = mlirOperationGetNextInBlock(op1);
        let op2 = mlirOperationGetNextInBlock(useOp1);
        let op3 = mlirOperationGetNextInBlock(op2);

        let rewriter = mlirIRRewriterCreate(ctx);

        // Test replace op with values
        let xz = [x, z];
        mlirRewriterBaseReplaceOpWithValues(rewriter, op1, 2, xz.as_ptr());

        // Test replace op with op
        mlirRewriterBaseReplaceOpWithOperation(rewriter, op2, op3);

        mlirOperationDump(op);
        // clang-format off
        // CHECK-NEXT: module {
        // CHECK-NEXT:   %[[res:.*]]:3 = "dialect.create_values"() : () -> (index, index, index)
        // CHECK-NEXT:   "dialect.use_op1"(%[[res]]#0, %[[res]]#2) : (index, index) -> ()
        // CHECK-NEXT:   %[[res2:.*]]:2 = "dialect.op3"() : () -> (index, index)
        // CHECK-NEXT:   "dialect.use_op2"(%[[res2]]#0, %[[res2]]#1) : (index, index) -> ()
        // CHECK-NEXT: }
        // clang-format on

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testErase(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testErase
        eprintln!("@testErase");

        let moduleString = "\"dialect.op_to_erase\"() : () -> ()
\"dialect.op2\"() ({
^bb0():
  \"dialect.op2_nested\"() : () -> ()
^block_to_erase():
  \"dialect.op2_nested\"() : () -> ()
^bb1():
  \"dialect.op2_nested\"() : () -> ()
}) : () -> ()\n\0"
            .as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        // get a handle to all operations/values
        let opToErase = mlirBlockGetFirstOperation(body);
        let op2 = mlirOperationGetNextInBlock(opToErase);
        let op2Region = mlirOperationGetRegion(op2, 0);
        let bb0 = mlirRegionGetFirstBlock(op2Region);
        let blockToErase = mlirBlockGetNextInRegion(bb0);

        let rewriter = mlirIRRewriterCreate(ctx);
        mlirRewriterBaseEraseOp(rewriter, opToErase);
        mlirRewriterBaseEraseBlock(rewriter, blockToErase);

        mlirOperationDump(op);
        // CHECK-NEXT: module {
        // CHECK-NEXT: "dialect.op2"() ({
        // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
        // CHECK-NEXT: ^{{.*}}:
        // CHECK-NEXT:   "dialect.op2_nested"() : () -> ()
        // CHECK-NEXT: }) : () -> ()
        // CHECK-NEXT: }

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testMove(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testMove
        eprintln!("@testMove");

        let moduleString = "\"dialect.op1\"() : () -> ()
\"dialect.op2\"() ({
^bb0(%arg0: index):
  \"dialect.op2_1\"(%arg0) : (index) -> ()
^bb1(%arg1: index):
  \"dialect.op2_2\"(%arg1) : (index) -> ()
}) : () -> ()
\"dialect.op3\"() : () -> ()
\"dialect.op4\"() : () -> ()\n\0"
            .as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        // get a handle to all operations/values
        let op1 = mlirBlockGetFirstOperation(body);
        let op2 = mlirOperationGetNextInBlock(op1);
        let op3 = mlirOperationGetNextInBlock(op2);
        let op4 = mlirOperationGetNextInBlock(op3);

        let region2 = mlirOperationGetRegion(op2, 0);
        let block0 = mlirRegionGetFirstBlock(region2);
        let block1 = mlirBlockGetNextInRegion(block0);

        // Test move operations.
        let rewriter = mlirIRRewriterCreate(ctx);
        mlirRewriterBaseMoveOpBefore(rewriter, op3, op1);
        mlirRewriterBaseMoveOpAfter(rewriter, op4, op1);
        mlirRewriterBaseMoveBlockBefore(rewriter, block1, block0);

        mlirOperationDump(op);
        // CHECK-NEXT: module {
        // CHECK-NEXT:   "dialect.op3"() : () -> ()
        // CHECK-NEXT:   "dialect.op1"() : () -> ()
        // CHECK-NEXT:   "dialect.op4"() : () -> ()
        // CHECK-NEXT:   "dialect.op2"() ({
        // CHECK-NEXT:   ^{{.*}}(%[[arg0:.*]]: index):
        // CHECK-NEXT:     "dialect.op2_2"(%[[arg0]]) : (index) -> ()
        // CHECK-NEXT:   ^{{.*}}(%[[arg1:.*]]: index):  // no predecessors
        // CHECK-NEXT:     "dialect.op2_1"(%[[arg1]]) : (index) -> ()
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT: }

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testOpModification(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testOpModification
        eprintln!("@testOpModification");

        let moduleString = "%x, %y = \"dialect.op1\"() : () -> (index, index)
\"dialect.op2\"(%x) : (index) -> ()\n\0"
            .as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        // get a handle to all operations/values
        let op1 = mlirBlockGetFirstOperation(body);
        let y = mlirOperationGetResult(op1, 1);
        let op2 = mlirOperationGetNextInBlock(op1);

        let rewriter = mlirIRRewriterCreate(ctx);
        mlirRewriterBaseStartOpModification(rewriter, op1);
        mlirRewriterBaseCancelOpModification(rewriter, op1);

        mlirRewriterBaseStartOpModification(rewriter, op2);
        mlirOperationSetOperand(op2, 0, y);
        mlirRewriterBaseFinalizeOpModification(rewriter, op2);

        mlirOperationDump(op);
        // CHECK-NEXT: module {
        // CHECK-NEXT: %[[xy:.*]]:2 = "dialect.op1"() : () -> (index, index)
        // CHECK-NEXT: "dialect.op2"(%[[xy]]#1) : (index) -> ()
        // CHECK-NEXT: }

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn testReplaceUses(ctx: MlirContext) {
    unsafe {
        // CHECK-LABEL: @testReplaceUses
        eprintln!("@testReplaceUses");

        let moduleString =
      // Replace values with values
      // Replace op with values
      // Replace op with op
      // Replace op in block;
      // Replace value with value except in op
      "%x1, %y1, %z1 = \"dialect.op1\"() : () -> (index, index, index)
%x2, %y2, %z2 = \"dialect.op2\"() : () -> (index, index, index)
\"dialect.op1_uses\"(%x1, %y1, %z1) : (index, index, index) -> ()
%x3 = \"dialect.op3\"() : () -> index
%x4 = \"dialect.op4\"() : () -> index
\"dialect.op3_uses\"(%x3) : (index) -> ()
%x5 = \"dialect.op5\"() : () -> index
%x6 = \"dialect.op6\"() : () -> index
\"dialect.op5_uses\"(%x5) : (index) -> ()
%x7 = \"dialect.op7\"() : () -> index
%x8 = \"dialect.op8\"() : () -> index
\"dialect.op9\"() ({
^bb0:
   \"dialect.op7_uses\"(%x7) : (index) -> ()
}): () -> ()
\"dialect.op7_uses\"(%x7) : (index) -> ()
%x10 = \"dialect.op10\"() : () -> index
%x11 = \"dialect.op11\"() : () -> index
\"dialect.op10_uses\"(%x10) : (index) -> ()
\"dialect.op10_uses\"(%x10) : (index) -> ()\n\0".as_ptr() as *const i8;

        let module = mlirModuleCreateParse(ctx, mlirStringRefCreateFromCString(moduleString));
        let op = mlirModuleGetOperation(module);
        let body = mlirModuleGetBody(module);

        // get a handle to all operations/values
        let op1 = mlirBlockGetFirstOperation(body);
        let x1 = mlirOperationGetResult(op1, 0);
        let y1 = mlirOperationGetResult(op1, 1);
        let z1 = mlirOperationGetResult(op1, 2);
        let op2 = mlirOperationGetNextInBlock(op1);
        let x2 = mlirOperationGetResult(op2, 0);
        let y2 = mlirOperationGetResult(op2, 1);
        let z2 = mlirOperationGetResult(op2, 2);
        let op1Uses = mlirOperationGetNextInBlock(op2);

        let op3 = mlirOperationGetNextInBlock(op1Uses);
        let op4 = mlirOperationGetNextInBlock(op3);
        let x4 = mlirOperationGetResult(op4, 0);
        let op3Uses = mlirOperationGetNextInBlock(op4);

        let op5 = mlirOperationGetNextInBlock(op3Uses);
        let op6 = mlirOperationGetNextInBlock(op5);
        let op5Uses = mlirOperationGetNextInBlock(op6);

        let op7 = mlirOperationGetNextInBlock(op5Uses);
        let op8 = mlirOperationGetNextInBlock(op7);
        let x8 = mlirOperationGetResult(op8, 0);
        let op9 = mlirOperationGetNextInBlock(op8);
        let region9 = mlirOperationGetRegion(op9, 0);
        let block9 = mlirRegionGetFirstBlock(region9);
        let op7Uses = mlirOperationGetNextInBlock(op9);

        let op10 = mlirOperationGetNextInBlock(op7Uses);
        let x10 = mlirOperationGetResult(op10, 0);
        let op11 = mlirOperationGetNextInBlock(op10);
        let x11 = mlirOperationGetResult(op11, 0);
        let op10Uses1 = mlirOperationGetNextInBlock(op11);

        let rewriter = mlirIRRewriterCreate(ctx);

        // Replace values
        mlirRewriterBaseReplaceAllUsesWith(rewriter, x1, x2);
        let y1z1 = [y1, z1];
        let y2z2 = [y2, z2];
        mlirRewriterBaseReplaceAllValueRangeUsesWith(rewriter, 2, y1z1.as_ptr(), y2z2.as_ptr());

        // Replace op with values
        mlirRewriterBaseReplaceOpWithValues(rewriter, op3, 1, &x4);

        // Replace op with op
        mlirRewriterBaseReplaceOpWithOperation(rewriter, op5, op6);

        // Replace op with op in block
        mlirRewriterBaseReplaceOpUsesWithinBlock(rewriter, op7, 1, &x8, block9);

        // Replace value with value except in op
        mlirRewriterBaseReplaceAllUsesExcept(rewriter, x10, x11, op10Uses1);

        mlirOperationDump(op);
        // clang-format off
        // CHECK-NEXT: module {
        // CHECK-NEXT:   %{{.*}}:3 = "dialect.op1"() : () -> (index, index, index)
        // CHECK-NEXT:   %[[res2:.*]]:3 = "dialect.op2"() : () -> (index, index, index)
        // CHECK-NEXT:   "dialect.op1_uses"(%[[res2]]#0, %[[res2]]#1, %[[res2]]#2) : (index, index, index) -> ()
        // CHECK-NEXT:   %[[res4:.*]] = "dialect.op4"() : () -> index
        // CHECK-NEXT:   "dialect.op3_uses"(%[[res4]]) : (index) -> ()
        // CHECK-NEXT:   %[[res6:.*]] = "dialect.op6"() : () -> index
        // CHECK-NEXT:   "dialect.op5_uses"(%[[res6]]) : (index) -> ()
        // CHECK-NEXT:   %[[res7:.*]] = "dialect.op7"() : () -> index
        // CHECK-NEXT:   %[[res8:.*]] = "dialect.op8"() : () -> index
        // CHECK-NEXT:   "dialect.op9"() ({
        // CHECK-NEXT:     "dialect.op7_uses"(%[[res8]]) : (index) -> ()
        // CHECK-NEXT:   }) : () -> ()
        // CHECK-NEXT:   "dialect.op7_uses"(%[[res7]]) : (index) -> ()
        // CHECK-NEXT:   %[[res10:.*]] = "dialect.op10"() : () -> index
        // CHECK-NEXT:   %[[res11:.*]] = "dialect.op11"() : () -> index
        // CHECK-NEXT:   "dialect.op10_uses"(%[[res10]]) : (index) -> ()
        // CHECK-NEXT:   "dialect.op10_uses"(%[[res11]]) : (index) -> ()
        // CHECK-NEXT: }
        // clang-format on

        mlirIRRewriterDestroy(rewriter);
        mlirModuleDestroy(module);
    }
}

fn main() {
    unsafe {
        let ctx = mlirContextCreate();
        mlirContextSetAllowUnregisteredDialects(ctx, 1);
        mlirContextGetOrLoadDialect(
            ctx,
            mlirStringRefCreateFromCString("builtin\0".as_ptr() as *const i8),
        );
        testInsertionPoint(ctx);
        testCreateBlock(ctx);
        testInlineRegionBlock(ctx);
        testReplaceOp(ctx);
        testErase(ctx);
        testMove(ctx);
        testOpModification(ctx);
        testReplaceUses(ctx);

        mlirContextDestroy(ctx);
    }
}
