
jprime: src/jprime.cc src/Graph.cc src/State.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc
	g++ -o jprime src/jprime.cc src/Graph.cc src/State.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc -Wall -std=c++11 -O3 -fno-rtti
