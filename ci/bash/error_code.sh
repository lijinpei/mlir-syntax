invalid_argument () {
  echo "invalid argument"
  caller
  exit 1
}

clang_not_found () {
  echo "clang not found"
  caller
  exit 2
}

failed_to_detect_mlir () {
  echo "failed to detect mlir"
  caller
  exit 3
}

not_implemented_yet() {
  echo "not implemented yet"
  caller
  exit 4
}
