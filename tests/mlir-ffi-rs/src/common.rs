#![allow(non_snake_case)]

use mlir::ExecutionEngine::*;
use mlir::Support::*;
use mlir::IR::*;

pub fn mlirLogicalResultIsSuccess(res: MlirLogicalResult) -> bool {
    return res.value != 0;
}
pub fn mlirLogicalResultIsFailure(res: MlirLogicalResult) -> bool {
    return res.value == 0;
}
pub fn mlirTypeIsNull(r#type: MlirType) -> bool {
    r#type.ptr == std::ptr::null_mut()
}
pub fn mlirExecutionEngineIsNull(jit: MlirExecutionEngine) -> bool {
    jit.ptr == std::ptr::null_mut()
}
pub fn mlirOperationIsNull(op: MlirOperation) -> bool {
    op.ptr == std::ptr::null_mut()
}
