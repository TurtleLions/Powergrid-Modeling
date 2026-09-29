# Power Grid engine (C++) and OpenSpiel game

`powergrid_core.py` in the repo root is the rules **spec**. It follows the
**Recharged** rulebook (`Power-Grid-Recharged-Rules.pdf`), including the
2-player "Against the Trust" rules, and plays on the Recharged German board by
default (the 2004 board is available as `germany-2004`). This directory is a C++ port of it for speed. The port
is checked move by move against the spec.

| File | What it is |
|---|---|
| `powergrid_engine.{h,cc}` | The engine: plain C++17 with no dependencies. The state is one flat struct of fixed arrays and 64-bit masks, so a clone is a ~650-byte copy. |
| `powergrid_data.inc` | Plants, both German boards and rule tables, **generated** from the spec by `../tools/gen_cpp_data.py`. Do not edit it by hand. |
| `powergrid_trace.cc` | Replays action sequences and prints canonical dumps, for the diff test. |
| `powergrid_bench.cc` | Random-play throughput and clone cost. |
| `openspiel/powergrid.{h,cc}` | OpenSpiel adapter that registers the game as `"powergrid"`. |
| `openspiel/powergrid_test.cc` | OpenSpiel's standard game tests plus a few specific ones. |
| `openspiel/install_into_openspiel.sh` | Adds the game to an OpenSpiel checkout. |
| `../tools/difftest.py` | Plays random games in Python and requires identical traces from C++. |

## Standalone (no OpenSpiel)

    make -C cpp          # build the trace tool and benchmark
    make -C cpp check    # data file up to date + diff test: 500 random games, C++ vs Python
    make -C cpp bench

## OpenSpiel

    cpp/openspiel/install_into_openspiel.sh /path/to/open_spiel   # symlinks + CMake patch
    cd /path/to/open_spiel/build && make -j powergrid_test pyspiel
    ./games/powergrid_test

From Python: `pyspiel.load_game("powergrid(players=4)")` is the rulebook game on
the German map, with the regions in play drawn at random. To fix them, use
`powergrid(players=3,regions=north-west+west+east)`. Two players get the Trust
unless `trust=0`. The parameters are documented in `openspiel/powergrid.h`.

## Changing a rule

1. Change `powergrid_core.py` and its tests (`python3 -m unittest`). Cite the
   Recharged rulebook page as `[R p.N]`.
2. If you changed a table, map or plant, run `python3 tools/gen_cpp_data.py`.
   Otherwise port the change to `powergrid_engine.cc`, keeping the Python
   function names.
3. Run `make -C cpp check` until the traces match again. If the change affects
   state, update `dump()` in `tools/difftest.py` and `State::Dump()` together.
