
jprime: src/jprime.cc src/jprime_tests.cc src/Graph.cc src/State.cc src/Worker.cc src/GenLoopsRecursive.cc src/GenLoopsIterative.cc src/Coordinator.cc src/WorkAssignment.cc src/SearchConfig.cc src/SearchContext.cc src/Pattern.cc
	g++ -o jprime src/jprime.cc src/jprime_tests.cc src/Graph.cc src/State.cc src/Worker.cc src/GenLoopsRecursive.cc src/GenLoopsIterative.cc src/Coordinator.cc src/WorkAssignment.cc src/SearchConfig.cc src/SearchContext.cc src/Pattern.cc -Wall -Wextra -std=c++20 -O3
