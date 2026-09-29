// powergrid_trace -- replays action sequences and prints the canonical trace
// that tools/difftest.py compares against the Python engine.
//
// stdin, per game:
//   game <players> <map> <play_regions> <regions|-> <step2_cities> <end_cities>
//        <max_rounds> <max_bid> <start_money> <houses> <trust>
//   (regions is a comma-separated list; -1 means "rulebook value")
//   <action> <action> ...          (one line, may be empty)
// stdout, per game: "D <dump>" for the initial state, then for every action
// "A <action string>" (player moves only), one "L <event>" per new log entry,
// and "D <dump>"; finally "END". An illegal action or a broken invariant
// prints "ERROR <what>" and skips the rest of that game.

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

#include "powergrid_engine.h"

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    std::istringstream hdr(line);
    std::string tag;
    auto rules = std::make_shared<powergrid::Rules>();
    std::string map_name, regions;
    hdr >> tag >> rules->num_players >> map_name >> rules->play_regions >> regions >>
        rules->step2_cities >> rules->end_cities >> rules->max_rounds >> rules->max_bid >>
        rules->start_money >> rules->houses >> rules->trust;
    if (tag != "game") {
      std::cerr << "bad header: " << line << "\n";
      return 2;
    }
    rules->map = powergrid::MapByName(map_name);
    if (regions != "-") {
      std::istringstream rs(regions);
      std::string tok;
      while (std::getline(rs, tok, ',')) rules->regions.push_back(std::stoi(tok));
    }
    rules->Finalize();
    std::getline(std::cin, line);
    std::istringstream acts(line);
    powergrid::State s(rules);
    std::cout << "D " << s.Dump() << "\n";
    size_t logged = 0;
    int a;
    try {
      while (acts >> a) {
        auto legal = s.LegalActions();
        if (std::find(legal.begin(), legal.end(), a) == legal.end())
          throw std::runtime_error("illegal action " + std::to_string(a));
        if (!s.IsChanceNode()) std::cout << "A " << s.ActionToString(a) << "\n";
        s.ApplyAction(a);
        s.CheckInvariants();
        for (; logged < s.log().size(); ++logged)
          std::cout << "L " << s.log()[logged].ToString() << "\n";
        std::cout << "D " << s.Dump() << "\n";
      }
    } catch (const std::exception& e) {
      std::cout << "ERROR " << e.what() << "\n";  // shows up as the first mismatch
    }
    std::cout << "END\n";
  }
  return 0;
}
