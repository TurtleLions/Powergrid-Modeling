// pgcore -- Python bindings for the C++ Power Grid engine, for bots and training.
//
// Independent of OpenSpiel: the engine alone, with a structured JSON view for
// scripted bots and observations as NumPy arrays for learning agents.
//
//   import pgcore
//   rules = pgcore.Rules(players=4)            # Recharged rules, German board
//   s = pgcore.State(rules)
//   while not s.is_terminal(): ...

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "../powergrid_engine.h"

namespace py = pybind11;
using powergrid::Rules;
using powergrid::State;

namespace {

std::shared_ptr<const Rules> MakeRules(int players, const std::string& map,
                                       const std::vector<int>& regions, int trust,
                                       int max_rounds, int max_bid, int start_money,
                                       bool keep_log) {
  auto r = std::make_shared<Rules>();
  r->num_players = players;
  r->map = powergrid::MapByName(map);
  r->regions = regions;
  r->trust = trust;
  r->max_rounds = max_rounds;
  r->max_bid = max_bid;
  r->start_money = start_money;
  r->keep_log = keep_log;
  r->Finalize();
  return r;
}

}  // namespace

PYBIND11_MODULE(pgcore, m) {
  m.doc() = "Power Grid (Recharged) engine";
  m.attr("CHANCE") = powergrid::kChancePlayer;
  m.attr("TERMINAL") = powergrid::kTerminalPlayer;
  m.attr("UNREACHABLE") = powergrid::kUnreachable;

  // Rules are immutable once made; State holds a shared pointer to them.
  py::class_<Rules, std::shared_ptr<Rules>>(m, "Rules")
      .def(py::init([](int players, const std::string& map, std::vector<int> regions, int trust,
                       int max_rounds, int max_bid, int start_money, bool keep_log) {
             return std::const_pointer_cast<Rules>(MakeRules(players, map, regions, trust,
                                                             max_rounds, max_bid, start_money,
                                                             keep_log));
           }),
           py::arg("players") = 4, py::arg("map") = "germany",
           py::arg("regions") = std::vector<int>{}, py::arg("trust") = 0,
           py::arg("max_rounds") = 100, py::arg("max_bid") = 400, py::arg("start_money") = 50,
           py::arg("keep_log") = false,
           "trust defaults to 0 (off); pass -1 for the rulebook default (on for 2 players).")
      .def_readonly("num_players", &Rules::num_players)
      .def_readonly("plant_limit", &Rules::plant_limit)
      .def_readonly("step2_cities", &Rules::step2_cities)
      .def_readonly("end_cities", &Rules::end_cities)
      .def_readonly("max_bid", &Rules::max_bid)
      .def_readonly("max_rounds", &Rules::max_rounds)
      .def_readonly("region_sets", &Rules::region_sets)
      .def_property_readonly("city_names", [](const Rules& r) { return r.map.city_names; })
      .def_property_readonly("city_region", [](const Rules& r) { return r.map.city_region; })
      .def_property_readonly("region_names", [](const Rules& r) { return r.map.region_names; })
      .def_property_readonly("num_cities", [](const Rules& r) { return r.map.num_cities; })
      .def_property_readonly("num_actions", [](const Rules& r) {
        return powergrid::Codec(r).num_actions;
      })
      .def_property_readonly("feature_size", [](const Rules& r) { return State::FeatureSize(r); })
      .def_property_readonly("observation_size",
                             [](const Rules& r) { return State::ObservationSize(r); })
      .def_property_readonly("codec", [](const Rules& r) {
        powergrid::Codec c(r);
        py::dict d;
        d["PASS"] = c.kPass;
        d["DONE"] = c.kDone;
        d["SELECT0"] = c.kSelect0;
        d["BID0"] = c.kBid0;
        d["BUY0"] = c.kBuy0;
        d["BUILD0"] = c.kBuild0;
        d["DISCARD0"] = c.kDiscard0;
        d["RUN0"] = c.kRun0;
        d["NUM_ACTIONS"] = c.num_actions;
        return d;
      });

  py::class_<State>(m, "State")
      .def(py::init([](std::shared_ptr<Rules> r) {
        return State(std::const_pointer_cast<const Rules>(r));
      }))
      .def("clone", [](const State& s) { return State(s); })
      .def("__copy__", [](const State& s) { return State(s); })
      .def("is_terminal", &State::IsTerminal)
      .def("is_chance_node", &State::IsChanceNode)
      .def("current_player", &State::CurrentPlayer)
      .def("legal_actions", &State::LegalActions)
      .def("chance_outcomes", &State::ChanceOutcomes)
      .def("apply_action", &State::ApplyAction)
      .def("returns", &State::Returns)
      .def("action_to_string", &State::ActionToString)
      .def("describe", &State::Describe)
      .def("to_json", &State::ToJson)
      .def("dump", &State::Dump)
      .def("check_invariants", &State::CheckInvariants)
      .def("build_cost", &State::BuildCost, "cost for player p to build in city now, -1 if not")
      .def("price", &State::Price, "price of the next unit of fuel f, -1 if sold out")
      .def("min_bid", &State::MinBid)
      .def("distance", &State::Distance, "cheapest connection cost between two cities in play")
      .def("max_supply", &State::MaxSupply,
           "most cities player p could power now with their plants and stored fuel")
      .def("round", &State::round)
      .def("step", &State::step)
      .def("money", &State::money)
      .def("observation",
           [](const State& s, int viewer) {
             py::array_t<float> out(State::ObservationSize(s.rules()));
             s.ObservationTensor(viewer, out.mutable_data());
             return out;
           })
      .def("features",
           [](const State& s, int seat) {
             py::array_t<float> out(State::FeatureSize(s.rules()));
             s.Features(seat, out.mutable_data());
             return out;
           },
           "bots/rl/features.py (version 2) features for seat, computed in C++")
      .def("features_all",
           [](const State& s) {
             const int n = s.num_players(), k = State::FeatureSize(s.rules());
             py::array_t<float> out({n, k});
             for (int p = 0; p < n; ++p) s.Features(p, out.mutable_data() + p * k);
             return out;
           },
           "features for every seat, shape [players, size]")
      .def("log", [](const State& s) {
        std::vector<std::string> out;
        for (const auto& e : s.log()) out.push_back(e.ToString());
        return out;
      });
}
