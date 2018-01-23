LD_FLAGS=-Lfaiss -lfaiss -lopenblas

CPP_FLAGS= -O3 -I. -g -fPIC
CPP_FLAGS+= -Wall -Wextra -Wno-unused-result
CPP_FLAGS+= -std=c++11 -fopenmp

BIND_FLAGS= -shared -fPIC
BIND_INCLUDES=$(shell python -m pybind11 --includes)
BIND_TARGET=bin/_pymips.so

COMPILE= g++ $(CPP_FLAGS)

SOURCES=$(shell find src -name \*.cpp)
OBJECTS=$(subst .cpp,.o,$(subst src,build,$(SOURCES)))
DEPS=$(subst .o,.d,$(OBJECTS))
BINDSRC= python/pybind/mips_wrapper.cpp python/pybind/wrapper_util.h

all: py

py: ext
	(python setup.py install)

ext: dirs $(BIND_TARGET)

dirs:
	(mkdir -p bin build data)

faiss/libfaiss.a:
	(cp makefile.inc faiss/makefile.inc)
	(cd faiss; make -j 4)

$(OBJECTS): build/%.o: src/%.cpp
	$(COMPILE) -MMD -MP -c $< -o $@

$(BIND_TARGET): faiss/libfaiss.a $(OBJECTS) $(BINDSRC)
	g++ $(BIND_FLAGS) $(CPP_FLAGS) $(BIND_INCLUDES) $(OBJECTS) $(BINDSRC) -o $(BIND_TARGET) $(LD_FLAGS)

clean:
	rm -rf bin build
	(cd faiss; make clean)


-include $(DEPS)
