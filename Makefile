TESTS = test_tree
TIMING_DIR = examples/timing/
TIMETESTS = merge_timing.py sample_timing.py pso_timing.py smt_query_timing.py query_timing.py
CFFI = _tree_cffi.*
CFFI_BUILD = cffi_build.py
INTERMEDIATE = tree.o test_tree.o
DIRS = __pycache__/ .pytest_cache/ Release/

CC = gcc
MAKE_EX = -o  # Make executables
MAKE_OBJ = -c  # Make object files
PY_DEBUG = # --debug  # Add --debug flag to compile C library with assertions
RUN_PYTEST = python -m pytest
RUN_CPROFILE = python -m cProfile

RM = rm -f
RMDIR = rm -r -f

all: tests cffi

# Clean Files
clean:
	$(RM) $(TESTS)

delDirectories:
	$(RMDIR) $(DIRS)

cleanPython: delDirectories
	$(RM) $(CFFI) *.pyd *.pstats

clobber: clean cleanPython
	$(RM) $(INTERMEDIATE)

# Run tests
pytest:
	$(RUN_PYTEST)

timetests:
	for test in $(TIMETESTS); do $(RUN_CPROFILE) -o _$$test.pstats $(TIMING_DIR)$$test; done;

ctests:
	for test in $(TESTS); do ./$$test; done

runtests: pytest ctests

# Build files
tests: $(TESTS)

cffi: $(CFFI_BUILD) tree.o cleanPython 
	python $(CFFI_BUILD) $(PY_DEBUG)

# Tests
test_tree: tree.o test_tree.o
	$(CC) tree.o test_tree.o $(MAKE_EX) test_tree -lm

# Object files
test_tree.o: test_tree.c tree.h
	$(CC) $(MAKE_OBJ) test_tree.c

tree.o: tree.c tree.h 
	$(CC) $(MAKE_OBJ) tree.c
