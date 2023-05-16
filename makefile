
jdeep: src/jdeep.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc
	g++ -o jdeep src/jdeep.cc src/Worker.cc src/Coordinator.cc src/WorkAssignment.cc -std=c++11 -O3 -fno-rtti
