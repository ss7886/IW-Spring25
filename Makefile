TESTS = testtree
CFFI = _tree_cffi.c
CFFI_BUILD = cffi_build.py
INTERMEDIATE = tree.o testtree.o
DIRS = __pycache__/ .pytest_cache/ Release/

CC = gcc
EXEC = -o  # Make executables
OBJ = -c  # Make object files
PICFLAG := $(if $(filter Unix,$(OS)),-fPIC,)  # Add -fPIC when compiling library on UNIX

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
	$(RM) $(INTERMEDIATE) $(CFFI)

# Run tests
pytest:
	pytest

runtests: pytest
	./$(TESTS)

# Build files
tests: $(TESTS)

cffi: $(CFFI_BUILD) tree.o cleanPython 
	python $(CFFI_BUILD)

int: $(INTERMEDIATE)

testtree: tree.o testtree.o
	$(CC) tree.o testtree.o $(EXEC) testtree -lm

testtree.o: testtree.c
	$(CC) $(OBJ) testtree.c

tree.o: tree.c tree.h 
	$(CC) $(OBJ) tree.c $(PICFLAG)
