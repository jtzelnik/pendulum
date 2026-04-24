# Control loop frequency in Hz.
# Override at build time: make LOOP_HZ=50
# Must match loop_hz in hardware/rl/config.yaml — changing it requires a rebuild.
LOOP_HZ  ?= 20

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -I. -I../common -D LOOP_HZ=$(LOOP_HZ)

# Sources shared by both binaries
COMMON_SRC := homing.cpp state.cpp

# ── Targets ───────────────────────────────────────────────────────────────────

.PHONY: all clean

all: lli manual_drive

# Main LLI binary — needs ZeroMQ in addition to pigpio
lli: lli_main.cpp lli_loop.cpp $(COMMON_SRC)
	@echo "Building lli at $(LOOP_HZ) Hz"
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpigpio -lpthread -lzmq

# Manual drive — no ZeroMQ
manual_drive: manual_drive.cpp $(COMMON_SRC)
	@echo "Building manual_drive at $(LOOP_HZ) Hz"
	$(CXX) $(CXXFLAGS) $^ -o $@ -lpigpio -lpthread

clean:
	rm -f lli manual_drive
