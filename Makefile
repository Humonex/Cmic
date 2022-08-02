CXX=g++
CPPFLAGS+=-Dunix
CXXFLAGS+=-O3
CXXFLAGS+=-std=c++11

all: cmic

libzpaq.o: libzpaq.cpp libzpaq.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c libzpaq.cpp

cmic.o: cmic.cpp cmic.h libzpaq.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -w -o $@ -c cmic.cpp -pthread

main.o: main.cpp cmic.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ -c main.cpp -pthread

cmic: main.o cmic.o libzpaq.o
	$(CXX) $(LDFLAGS) -o $@ main.o cmic.o libzpaq.o -pthread

clean:
	rm -f main.o cmic.o libzpaq.o cmic
