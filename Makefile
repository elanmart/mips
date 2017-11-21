LD_FLAGS=-Lfaiss -lfaiss -lopenblas

CPP_FLAGS= -O3 -I. -g -fPIC
CPP_FLAGS+= -Wall -Wextra -Wno-unused-result
CPP_FLAGS+= -std=c++11 -fopenmp

BIND_FLAGS= -shared -fPIC
BIND_INCLUDES=$(shell python3 -m pybind11 --includes)
BIND_TARGET=python/mips.so

COMPILE= g++ $(CPP_FLAGS)

SOURCES=$(shell find src -name \*.cpp)
OBJECTS=$(subst .cpp,.o,$(subst src,build,$(SOURCES)))
DEPS=$(subst .o,.d,$(OBJECTS))

TESTSRC=$(shell find tests -name \*.cpp)
TESTBINS=$(subst .cpp,,$(subst tests,bin,$(TESTSRC)))

BINDSRC=$(shell find wrap -name \*.cpp)
BINDSRC+=$(shell find wrap -name \*.h)

all: dirs $(TESTBINS)

$(OBJECTS): build/%.o: src/%.cpp
	$(COMPILE) -MMD -MP -c $< -o $@

bin/%: tests/%.cpp faiss/libfaiss.a $(OBJECTS)
	$(COMPILE) $^ -o $@ $(LD_FLAGS)

faiss/libfaiss.a:
	(cp makefile.inc faiss/makefile.inc)
	(cd faiss; make -j 4)

dirs:
	(mkdir -p bin build data)

clean:
	rm -rf bin build
	rm -f $(BIND_TARGET)
	(cd faiss; make clean)

pyfaiss: faiss/swigfaiss.swig
	(cd faiss; make py -j 4)

$(BIND_TARGET): faiss/libfaiss.a $(OBJECTS) $(BINDSRC)
	g++ $(BIND_FLAGS) $(CPP_FLAGS) $(BIND_INCLUDES) $(OBJECTS) $(BINDSRC) -o $(BIND_TARGET) $(LD_FLAGS)

.PHONY: bind
bind: $(BIND_TARGET)



-include $(DEPS)
