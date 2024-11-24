configure_file("${SRC}" "${DEST}.tmp" @ONLY)
file(COPY_FILE "${DEST}.tmp" "${DEST}" ONLY_IF_DIFFERENT)
