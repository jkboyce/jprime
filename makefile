
jprime: src/jprime.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc
	g++ -o jprime src/jprime.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc -std=c++11 -O3 -fno-rtti
