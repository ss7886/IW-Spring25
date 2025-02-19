# Detect OS (works for Windows and Unix)
OS := $(shell uname 2>/dev/null | grep -i mingw > /dev/null && echo Windows || echo Unix)

# Define extension for C-library (.dll for Windows, .so for Unix)
EXT := $(if $(filter Windows,$(OS)),dll,so)

TESTS = testtree
INTERMEDIATE = tree.o testtree.o
LIBS = tree.$(EXT)

CC = gcc
EXEC = -o  # Make executables
OBJ = -c  # Make object files
LIB = -shared -o  # Make libraries

RM = rm -f

all: tests libs

clean:
	$(RM) $(TESTS) $(LIBS)

clobber: clean
	$(RM) $(INTERMEDIATE)

tests: $(TESTS)

libs: $(LIBS)

int: $(INTERMEDIATE)

testtree: tree.o testtree.o
	$(CC) tree.o testtree.o $(EXEC) testtree -lm

tree.$(EXT): tree.o
	$(CC) $(LIB) tree.$(EXT) tree.o -lm

testtree.o: testtree.c
	$(CC) $(OBJ) testtree.c

tree.o: tree.c tree.h
	$(CC) $(OBJ) tree.c
