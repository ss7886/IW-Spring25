TESTS = test_tree
CFFI = _tree_cffi.*
CFFI_BUILD = cffi_build.py
INTERMEDIATE = tree.o test_tree.o
DIRS = __pycache__/ .pytest_cache/ Release/

CC = gcc
EXEC = -o  # Make executables
OBJ = -c  # Make object files
PICFLAG := $(if $(filter Unix,$(OS)),-fPIC,)  # Add -fPIC when compiling library on UNIX
PY_DEBUG = --debug  # Import C library with assertions

RM = rm -f
RMDIR = rm -r -f

all: tests cffi

# Clean Files
clean:
	$(RM) $(TESTS)

delDirectories:
	$(RMDIR) $(DIRS)

cleanPython: delDirectories
	$(RM) $(CFFI) *.pyd

clobber: clean cleanPython
	$(RM) $(INTERMEDIATE)

# Run tests
pytest:
	pytest

ctests:
	./$(TESTS)

runtests: pytest ctests

# Build files
tests: $(TESTS)

cffi: $(CFFI_BUILD) tree.o cleanPython 
	python $(CFFI_BUILD) $(PY_DEBUG)

int: $(INTERMEDIATE)

test_tree: tree.o test_tree.o
	$(CC) tree.o test_tree.o $(EXEC) test_tree -lm

test_tree.o: test_tree.c
	$(CC) $(OBJ) test_tree.c

tree.o: tree.c tree.h 
	$(CC) $(OBJ) tree.c $(PICFLAG)
