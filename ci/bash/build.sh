set -exo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/error_code.sh
source ${SCRIPT_DIR}/common.sh

if [ "$#" -lt 1 ];
then
  set -- build
fi

do_action () {
  if [ "$#" -lt 1 ] ; then
    invalid_argument
  fi
  action="$1"
  shift
  source ${SCRIPT_DIR}/do_"$action".sh "$@"
}

do_action "$@"
