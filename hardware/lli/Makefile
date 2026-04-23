CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -I. -I../common

# Sources shared by both binaries
COMMON_SRC := homing.cpp state.cpp

# ── Targets ───────────────────────────────────────────────────────────────────

.PHONY: all clean

all: lli manual_drive

# Main LLI binary — needs ZeroMQ in addition to pigpio
lli: lli_main.cpp lli_loop.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpigpio -lpthread -lzmq

# Manual drive — no ZeroMQ
manual_drive: manual_drive.cpp $(COMMON_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpigpio -lpthread

clean:
	rm -f lli manual_drive
