if [ ! -z "$_detected_clang" ]
then
  return
fi
_detected_clang=true
if command -v clang >/dev/null 2>&1
then
  CLANG_CC=clang
  CLANG_CXX=clang++
else
  for suffix in 18 19 20;
  do
    if command -v clang-${suffix} >/dev/null 2>&1
    then
      CLANG_CC=clang-${suffix}
      CLANG_CXX=clang++-${suffix}
    fi
  done
  if [ -z "${CLANG_CC}" ]
  then
    clang_not_found
  fi
fi
echo "detected clang ${CLANG_CC} ${CLANG_CXX}"
