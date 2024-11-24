#![allow(non_snake_case)]

use mlir_capi::ExecutionEngine::*;
use mlir_capi::Support::*;
use mlir_capi::IR::*;

pub fn mlirLogicalResultIsSuccess(res: MlirLogicalResult) -> bool {
    res.value != 0
}
pub fn mlirLogicalResultIsFailure(res: MlirLogicalResult) -> bool {
    res.value == 0
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

#[macro_export]
macro_rules! c_str {
    ($($s: literal)*) => { concat!($($s),*, '\0') };
}

#[macro_export]
macro_rules! c_str_ptr {
    ($($s: literal)*) => { $crate::c_str!($($s)*).as_ptr() as *const std::ffi::c_char };
}
