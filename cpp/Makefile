# Standalone build of the engine, trace tool and benchmark (no OpenSpiel needed).
CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -g -Wall -Wextra
BIN := build

all: $(BIN)/powergrid_trace $(BIN)/powergrid_bench

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

.PHONY: all check bench clean
