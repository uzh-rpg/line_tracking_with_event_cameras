# pragma once

struct Event {
  int x;
  int y;
  double t;
  bool p;
};

struct greater_than
{
  inline bool operator() (const Event & ev_1, const Event &ev_2)
  {
    return ev_1.t > ev_2.t;
  }
};