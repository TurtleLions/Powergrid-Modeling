// Features for the learning agents: a C++ port of Encoder._encode in
// bots/rl/features.py (version 2), value for value. Arithmetic is done in
// double and rounded to float at the end, as NumPy does, so the result is
// bit-identical to the Python encoder (checked by bots/test_bots.py).

#include <algorithm>
#include <vector>

#include "powergrid_engine.h"

namespace powergrid {

namespace {

constexpr int kPhases = 10;      // features.PHASES
constexpr int kPlantFeats = 1 + kNumKinds + 2;  // number, kind one-hot, need, power

struct Out {
  float* p;
  void add(double x) { *p++ = static_cast<float>(x); }
  void zeros(int k) { for (int i = 0; i < k; ++i) *p++ = 0.0f; }
};

void AddPlant(Out* o, const Plant* pl) {
  if (!pl) { o->zeros(kPlantFeats); return; }
  o->add(pl->number / 50.0);
  for (int k = 0; k < kNumKinds; ++k) o->add(pl->kind == k ? 1.0 : 0.0);
  o->add(pl->need / 3.0);
  o->add(pl->power / 7.0);
}

}  // namespace

int State::FeatureSize(const Rules& r) {
  const int slots = r.plant_limit + 1;
  return 27 + kMarketSlots * (kPlantFeats + 3) + kMaxPlayers * (6 + 4 + slots * kPlantFeats) +
         (1 + kPlantFeats + 1 + kMaxPlayers + 2) + r.map.num_cities * 5 + (2 * kMaxPlayers + 3);
}

void State::Features(int seat, float* out) const {
  const Rules& r = *rules_;
  const int n = n_;
  Out o{out};
  // --- global
  o.add(d_.round / 30.0);
  for (int s = 1; s <= 3; ++s) o.add(d_.step == s ? 1.0 : 0.0);
  for (int ph = 0; ph < kPhases; ++ph) o.add(static_cast<int>(d_.phase) == ph ? 1.0 : 0.0);
  o.add(d_.step3_pending ? 1.0 : 0.0);
  o.add(d_.uranium_stopped ? 1.0 : 0.0);
  o.add((d_.real_plug + d_.real_socket + (d_.top_pending ? 1 : 0)) / 42.0);
  o.add(d_.top_pending ? 1.0 : 0.0);
  for (int f = 0; f < 3; ++f) o.add(d_.fuel_market[f] / 24.0);
  o.add(d_.fuel_market[3] / 12.0);
  for (int f = 0; f < kNumFuels; ++f) {
    int pr = Price(f);
    o.add(pr >= 0 ? pr / 16.0 : 1.5);
  }
  o.add(static_cast<double>(n) / kMaxPlayers);
  // --- plant market (ascending), "current" = among the purchasable ones
  std::vector<int> market;
  for (int num = 0; num <= kMaxPlantNumber; ++num)
    if ((d_.market >> num) & 1) market.push_back(num);
  const int buyable = static_cast<int>(purchasable().size());
  for (int k = 0; k < kMarketSlots; ++k) {
    if (k < static_cast<int>(market.size())) {
      const int num = market[k];
      AddPlant(&o, &plant(num));
      o.add(k < buyable ? 1.0 : 0.0);
      o.add(d_.discount && num == d_.discount ? 1.0 : 0.0);
      o.add(MinBid(num) / 50.0);
    } else {
      AddPlant(&o, nullptr);
      o.zeros(3);
    }
  }
  // --- players, acting seat first, padded to kMaxPlayers
  const int slots = r.plant_limit + 1;
  for (int i = 0; i < kMaxPlayers; ++i) {
    if (i >= n) { o.zeros(6 + 4 + slots * kPlantFeats); continue; }
    const int p = (seat + i) % n;
    int cap = 0;
    for (int j = 0; j < d_.num_plants[p]; ++j) cap += plant(d_.plants[p][j]).power;
    int order_pos = 0;
    for (int j = 0; j < n; ++j) if (d_.order[j] == p) { order_pos = j; break; }
    o.add(1.0);
    o.add(d_.money[p] / 100.0);
    o.add(num_cities_of(p) / 20.0);
    o.add(cap / 20.0);
    o.add(static_cast<double>(order_pos) / kMaxPlayers);
    o.add((d_.done_auction >> p) & 1 ? 1.0 : 0.0);
    for (int f = 0; f < kNumFuels; ++f) o.add(d_.stored[p][f] / 10.0);
    for (int j = 0; j < slots; ++j)
      AddPlant(&o, j < d_.num_plants[p] ? &plant(d_.plants[p][j]) : nullptr);
  }
  // --- auction
  if (d_.auction_active) {
    const Auction& au = d_.auction;
    const bool in_market = au.plant >= 0 && ((d_.market >> au.plant) & 1);
    const int high = au.high < 0 ? -1 : ((au.high - seat) % n + n) % n;
    bool in_ring = false;
    for (int j = 0; j < au.ring_len; ++j) in_ring |= au.ring[j] == seat;
    o.add(1.0);
    AddPlant(&o, in_market ? &plant(au.plant) : nullptr);
    o.add(au.bid / 100.0);
    for (int i = 0; i < kMaxPlayers; ++i) o.add(high == i ? 1.0 : 0.0);
    o.add(in_ring ? 1.0 : 0.0);
    o.add(static_cast<double>(au.ring_len) / kMaxPlayers);
  } else {
    o.zeros(1 + kPlantFeats + 1 + kMaxPlayers + 2);
  }
  // --- cities: in play, occupancy, and what building there costs us now
  std::vector<bool> region_in(r.map.num_regions, false);
  if (d_.region_set >= 0)
    for (int g : r.region_sets[d_.region_set]) region_in[g] = true;
  for (int c = 0; c < r.map.num_cities; ++c) {
    const int k = d_.num_occupants[c];
    int mine = 0, others = 0;
    for (int j = 0; j < k; ++j) {
      const int q = d_.occupants[c][j];
      if (q == seat) mine = 1;
      else if (q < n) ++others;
    }
    const int cost = BuildCost(seat, c);
    o.add(region_in[r.map.city_region[c]] ? 1.0 : 0.0);
    o.add(k / 3.0);
    o.add(mine ? 1.0 : 0.0);
    o.add(others / 3.0);
    o.add(cost >= 0 ? cost / 50.0 : -1.0);
  }
  // --- end game: who would win if it ended now, and how close the end is
  const int end = r.end_cities;
  std::vector<int> supply(n);
  for (int p = 0; p < n; ++p) supply[p] = MaxSupply(p);
  for (int i = 0; i < kMaxPlayers; ++i) {
    if (i >= n) { o.zeros(2); continue; }
    const int p = (seat + i) % n;
    o.add(supply[p] / 20.0);
    o.add(static_cast<double>(end - num_cities_of(p)) / end);
  }
  std::pair<int, int> best_other{-1, -1};
  bool first = true;
  for (int p = 0; p < n; ++p) {
    if (p == seat) continue;
    std::pair<int, int> v{supply[p], d_.money[p]};
    if (first || v > best_other) { best_other = v; first = false; }
  }
  o.add((supply[seat] - best_other.first) / 10.0);
  o.add(std::make_pair(supply[seat], static_cast<int>(d_.money[seat])) > best_other ? 1.0 : 0.0);
  o.add(num_cities_of(seat) + 1 >= end ? 1.0 : 0.0);
}

}  // namespace powergrid
