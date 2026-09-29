#!/bin/bash
# Adds the Power Grid game to an OpenSpiel source checkout.
#
#   cpp/openspiel/install_into_openspiel.sh /path/to/open_spiel
#
# - symlinks the engine and adapter sources into open_spiel/games/powergrid/
#   (edits in this repo are picked up on the next OpenSpiel build)
# - registers the sources and the powergrid_test target in games/CMakeLists.txt
# - adds "powergrid" to the expected game list in python/tests/pyspiel_test.py
# Safe to run more than once.
set -euo pipefail

OS_ROOT=$(realpath "${1:?usage: $0 /path/to/open_spiel}")
HERE=$(dirname "$(realpath "$0")")
ENGINE_DIR=$(dirname "$HERE")
GAMES="$OS_ROOT/open_spiel/games"
[[ -f "$GAMES/CMakeLists.txt" ]] || { echo "not an OpenSpiel checkout: $OS_ROOT" >&2; exit 1; }

mkdir -p "$GAMES/powergrid"
for f in "$ENGINE_DIR/powergrid_engine.h" "$ENGINE_DIR/powergrid_engine.cc" "$ENGINE_DIR/powergrid_data.inc" \
         "$HERE/powergrid.h" "$HERE/powergrid.cc" "$HERE/powergrid_test.cc"; do
  ln -sfn "$f" "$GAMES/powergrid/$(basename "$f")"
done

python3 - "$GAMES/CMakeLists.txt" "$OS_ROOT/open_spiel/python/tests/pyspiel_test.py" <<'EOF'
import re
import sys

cmake_path, pyspiel_test_path = sys.argv[1:3]

cmake = open(cmake_path).read()
if "powergrid/powergrid_data.inc" not in cmake and "powergrid/powergrid.cc" in cmake:
    cmake = cmake.replace("  powergrid/powergrid_engine.h\n",
                          "  powergrid/powergrid_engine.h\n  powergrid/powergrid_data.inc\n", 1)
    open(cmake_path, "w").write(cmake)
    print("patched", cmake_path, "(data file)")
if "powergrid/powergrid.cc" not in cmake:
    sources = ("  powergrid/powergrid.cc\n  powergrid/powergrid.h\n"
               "  powergrid/powergrid_engine.cc\n  powergrid/powergrid_engine.h\n"
               "  powergrid/powergrid_data.inc\n")
    cmake = cmake.replace("set(GAME_SOURCES\n", "set(GAME_SOURCES\n" + sources, 1)
    cmake += ("\nadd_executable(powergrid_test powergrid/powergrid_test.cc ${OPEN_SPIEL_OBJECTS}\n"
              "               $<TARGET_OBJECTS:tests>)\n"
              "add_test(powergrid_test powergrid_test)\n")
    open(cmake_path, "w").write(cmake)
    print("patched", cmake_path)

test = open(pyspiel_test_path).read()
if '"powergrid",' not in test:
    # insert into the alphabetical list of expected registered games
    names = re.findall(r'^    "([a-z0-9_]+)",$', test, flags=re.M)
    after = max(n for n in names if n < "powergrid")
    test = test.replace(f'    "{after}",\n', f'    "{after}",\n    "powergrid",\n', 1)
    open(pyspiel_test_path, "w").write(test)
    print("patched", pyspiel_test_path)
EOF
echo "powergrid installed into $OS_ROOT"
