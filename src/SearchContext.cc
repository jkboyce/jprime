//
// SearchContext.cc
//
// Methods for saving and loading SearchContext structures on disk.
//
// Copyright (C) 1998-2024 Jack Boyce, <jboyce@gmail.com>
//
// This file is distributed under the MIT License.
//

#include "SearchContext.h"

#include <fstream>
#include <iomanip>
#include <thread>
#include <algorithm>
#include <stdexcept>


//------------------------------------------------------------------------------
// Sorting patterns
//------------------------------------------------------------------------------

// Convert a pattern line printed as comma-separated integers into a vector of
// ints.

std::vector<int> parse_pattern_line(const std::string& p) {
  // remove the first colon and anything beyond
  std::string pat = {p.cbegin(), std::find(p.cbegin(), p.cend(), ':')};

  // extract part between first and second '*'s if present
  const auto star = std::find(pat.cbegin(), pat.cend(), '*');
  if (star != pat.cend()) {
    pat = {star + 1, std::find(star + 1, pat.cend(), '*')};
  }

  std::vector<int> result;
  auto x = pat.cbegin();
  while (true) {
    auto y = std::find(x, pat.cend(), ',');
    std::string s{x, y};
    result.push_back(std::stoi(s));
    if (y == pat.cend())
      break;
    x = y + 1;
  }

  return result;
}

// Compare patterns printed as comma-separated integers
//
// Test case: jprime 7 42 6 -file test

bool pattern_compare_ints(const std::string& pat1, const std::string& pat2) {
  std::vector<int> vec1 = parse_pattern_line(pat1);
  std::vector<int> vec2 = parse_pattern_line(pat2);

  if (vec2.size() == 0)
    return false;
  if (vec1.size() == 0)
    return true;

  // shorter patterns sort earlier
  if (vec1.size() < vec2.size())
    return true;
  if (vec1.size() > vec2.size())
    return false;

  // ground state before excited state patterns
  if (pat1[0] == ' ' && pat2[0] == '*')
    return true;
  if (pat1[0] == '*' && pat2[0] == ' ')
    return false;

  // sort lower leading throws first
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] == vec2[i])
      continue;
    return vec1[i] < vec2[i];
  }
  return false;
}

// Compare patterns printed with letters (10='a', etc.)

bool pattern_compare_letters(const std::string& pat1, const std::string& pat2) {
  if (pat2.size() == 0)
    return false;
  if (pat1.size() == 0)
    return true;

  unsigned pat1_start = (pat1[0] == ' ' || pat1[0] == '*') ? 2 : 0;
  unsigned pat1_end = pat1_start;
  while (pat1_end != pat1.size() && pat1[pat1_end] != ' ') {
    ++pat1_end;
  }

  unsigned pat2_start = (pat2[0] == ' ' || pat2[0] == '*') ? 2 : 0;
  unsigned pat2_end = pat2_start;
  while (pat2_end != pat2.size() && pat2[pat2_end] != ' ') {
    ++pat2_end;
  }

  // shorter patterns sort earlier
  if (pat1_end - pat1_start < pat2_end - pat2_start)
    return true;
  if (pat1_end - pat1_start > pat2_end - pat2_start)
    return false;

  // ground state before excited state patterns
  if (pat1[0] == ' ' && pat2[0] == '*')
    return true;
  if (pat1[0] == '*' && pat2[0] == ' ')
    return false;

  // ascii order, except '+' is higher than any other character
  for (size_t i = pat1_start; i < pat1_end; ++i) {
    if (pat1[i] == pat2[i])
      continue;
    if (pat1[i] == '+' && pat2[i] != '+')
      return false;
    if (pat1[i] != '+' && pat2[i] == '+')
      return true;
    return pat1[i] < pat2[i];
  }
  return false;
}

// Standard library compliant Compare relation for patterns.
//
// Returns true if the first argument appears before the second in a strict
// weak ordering, and false otherwise.

bool pattern_compare(const std::string& pat1, const std::string& pat2) {
  bool has_comma = (std::find(pat1.cbegin(), pat1.cend(), ',') != pat1.cend() ||
      std::find(pat2.cbegin(), pat2.cend(), ',') != pat2.cend());

  if (has_comma) {
    return pattern_compare_ints(pat1, pat2);
  } else {
    return pattern_compare_letters(pat1, pat2);
  }
}

//------------------------------------------------------------------------------
// Saving checkpoint files
//------------------------------------------------------------------------------

void SearchContext::to_file(const std::string& file) {
  std::sort(patterns.begin(), patterns.end(), pattern_compare);

  std::ofstream myfile;
  myfile.open(file, std::ios::out | std::ios::trunc);
  if (!myfile || !myfile.is_open())
    return;

  myfile << "version           6.9\n"
         << "command line      " << arglist << '\n'
         << "states            " << full_numstates << '\n'
         << "shift cycles      " << full_numcycles << '\n'
         << "short cycles      " << full_numshortcycles << '\n'
         << "period bound      " << n_bound << '\n'
         << "states (in mem)   " << memory_numstates << '\n'
         << "patterns          " << npatterns << '\n'
         << "patterns (total)  " << ntotal << '\n'
         << "nodes completed   " << nnodes << '\n'
         << "work splits       " << splits_total << '\n'
         << "seconds elapsed   " << std::fixed << std::setprecision(4)
                                 << secs_elapsed << '\n'
         << "seconds working   " << secs_working << '\n'
         << "seconds avail     " << secs_available << '\n'
         << "cores avail       " << std::thread::hardware_concurrency() << '\n';

  myfile << "\npatterns\n";
  for (const std::string& str : patterns) {
    myfile << str << '\n';
  }

  myfile << "\ncounts\n";
  for (size_t i = 1; i < count.size(); ++i) {
    myfile << i << ", " << count.at(i) << '\n';
  }

  if (assignments.size() > 0) {
    myfile << "\nwork assignments remaining\n";
    for (const WorkAssignment& wa : assignments) {
      myfile << "  " << wa << '\n';
    }
  }

  myfile.close();
}

//------------------------------------------------------------------------------
// Loading checkpoint files
//------------------------------------------------------------------------------

static inline void trim(std::string& s) {
  // trim left, then trim right
  s.erase(
      s.begin(),
      std::find_if(s.begin(), s.end(),
          [](unsigned char ch) { return !std::isspace(ch); })
  );
  s.erase(
      std::find_if(s.rbegin(), s.rend(),
          [](unsigned char ch) { return !std::isspace(ch); }).base(),
      s.end()
  );
}

// Load a checkpoint file from disk into the SearchContext structure.
//
// In the event of an error, throw a `std::invalid_argument` exception with a
// relevant error message.

void SearchContext::from_file(const std::string& file) {
  std::ifstream myfile;
  myfile.open(file, std::ios::in);
  if (!myfile || !myfile.is_open()) {
    throw std::invalid_argument("Error reading file: could not open");
  }

  std::string version;
  std::string s;
  int linenum = 0;
  int section = 1;
  const int column_start = 17;

  while (myfile) {
    std::getline(myfile, s);
    std::string val;
    std::string error;

    switch (linenum) {
      case 0:
        if (s.rfind("version", 0) != 0) {
          error = "syntax in line 1";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        if (val != "6.8" && val != "6.9") {
          error = "file version below 6.8 not supported";
        }
        version = val;
        break;
      case 1:
        if (s.rfind("command", 0) != 0) {
          error = "syntax in line 2";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        arglist = val;
        break;
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
        break;
      case 7:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 8";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        npatterns = std::stoull(val);
        break;
      case 8:
        if (s.rfind("patterns (", 0) != 0) {
          error = "syntax in line 9";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        ntotal = std::stoull(val);
        break;
      case 9:
        if (s.rfind("nodes", 0) != 0) {
          error = "syntax in line 10";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        nnodes = std::stoull(val);
        break;
      case 10:
        if (version == "6.8") {
          ++linenum;  // skip and read line as "seconds elapsed" instead
          [[fallthrough]];
        } else {
          if (s.rfind("work splits", 0) != 0) {
            error = "syntax in line 11";
            break;
          }
          val = s.substr(column_start, s.size());
          trim(val);
          splits_total = static_cast<unsigned>(std::stoi(val));
          break;
        }
      case 11:
        if (s.rfind("seconds elapsed", 0) != 0) {
          error = "syntax in line 12";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        secs_elapsed = std::stod(val);
        break;
      case 12:
        if (s.rfind("seconds working", 0) != 0) {
          error = "syntax in line 13";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        secs_working = std::stod(val);
        break;
      case 13:
        if (s.rfind("seconds avail", 0) != 0) {
          error = "syntax in line 14";
          break;
        }
        val = s.substr(column_start, s.size());
        trim(val);
        secs_available = std::stod(val);
        break;
      case 14:
      case 15:
        break;
      case 16:
        if (s.rfind("patterns", 0) != 0) {
          error = "syntax in line 17";
          break;
        }
        section = 2;
        break;
    }

    if (error.size() > 0) {
      myfile.close();
      std::string msg("Error reading file: ");
      msg.append(error);
      throw std::invalid_argument(msg);
    }

    if (linenum < 17) {
      ++linenum;
      continue;
    }

    val = s;
    trim(val);

    if (val.size() == 0) {
      // ignore empty lines
    } else if (s.rfind("counts", 0) == 0) {
      section = 3;
    } else if (s.rfind("work", 0) == 0) {
      section = 4;
    } else if (section == 2) {
      patterns.push_back(s);
    } else if (section == 3) {
      // read counts
      size_t commapos = s.find(",");
      if (commapos == std::string::npos) {
        myfile.close();
        std::string msg("Error reading count in line ");
        msg.append(std::to_string(linenum + 1));
        throw std::invalid_argument(msg);
      }
      std::string field1 = s.substr(0, commapos);
      std::string field2 = s.substr(commapos + 1, s.size());
      int i = std::stoi(field1);
      count.resize(i + 1, 0);
      count.at(i) = std::stoull(field2);
    } else if (section == 4) {
      // read work assignments
      WorkAssignment wa;
      if (wa.from_string(val)) {
        assignments.push_back(wa);
      } else {
        myfile.close();
        std::string msg("Error reading work assignment in line ");
        msg.append(std::to_string(linenum + 1));
        throw std::invalid_argument(msg);
      }
    }

    ++linenum;
  }
  myfile.close();
}
