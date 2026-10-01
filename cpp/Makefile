# Standalone build of the engine, trace tool and benchmark (no OpenSpiel needed).
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra
BIN := build

all: $(BIN)/powergrid_trace $(BIN)/powergrid_bench

# Python bindings (pgcore) for bots and training; needs pybind11 and Python headers.
PYTHON ?= $(HOME)/.local/opt/python311/bin/python3
PGCORE := $(BIN)/pgcore$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

pgcore: $(PGCORE)

$(PGCORE): python/pgcore.cc powergrid_engine.cc powergrid_features.cc powergrid_engine.h powergrid_data.inc | $(BIN)
	$(CXX) -std=c++17 -O3 -g -Wall -Wextra -shared -fPIC \
	  $(shell $(PYTHON) -m pybind11 --includes) -o $@ python/pgcore.cc powergrid_engine.cc powergrid_features.cc

$(BIN)/%: %.cc powergrid_engine.cc powergrid_engine.h powergrid_data.inc | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $< powergrid_engine.cc

$(BIN):
	mkdir -p $(BIN)

# Engine must match powergrid_core.py move for move.
check: all
	python3 ../tools/gen_cpp_data.py --check
	python3 ../tools/difftest.py 500

bench: $(BIN)/powergrid_bench
	$(BIN)/powergrid_bench 20000 4 0 germany

clean:
	rm -rf $(BIN)

.PHONY: all pgcore check bench clean
