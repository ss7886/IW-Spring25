from argparse import ArgumentParser
import os
import sys

from cffi import FFI

def read_header(path: str) -> str:
    """
    Read and format C header file to use as source file for CFFI.

        path:           Path to C header. Include extension (e.g. "tree.h").
    """
    string = ""
    with open(path, 'r') as file:
        for line in file:
            if "#" in line or line.startswith("/*"):
                continue

            if line.startswith("EXPORT_SYMBOL "):
                string += line[14:]
            else:
                string += line
    return string

def compile_library(output_path: str, src_path: str, header_path: str,
                    debug: bool = False):
    """
    Return CFFI library.

        output_path:    Name of output file.
        lib_path:       Path to shared C library. Don't include extension
                        ("./tree" instead of "./tree.dll"). Extension is 
                        automatically added based on operating system.
        header_path:    Path to C header. Include extension (e.g. "tree.h").
    """
    ffi = FFI()

    ffi.cdef(read_header(header_path))
    extra_args = []
    if debug:
        extra_args.append('-U')
        extra_args.append('NDEBUG')
    ffi.set_source(output_path, f'#include "{header_path}"',
                   sources=[src_path], extra_compile_args=extra_args)
    ffi.compile()

def handle_args():
    """
    Handle and return arguments using ArgumentParser.
    """
    parser = ArgumentParser(prog=sys.argv[0],
                            description="Build CFFI.",
                            allow_abbrev=False)
    parser.add_argument("-d", "--debug", action="store_true",
                        help="the host on which the server is running")
    args = vars(parser.parse_args())
    return args["debug"]

if __name__ == "__main__":
    debug = handle_args()
    lib = compile_library("_tree_cffi", "tree.c", "tree.h", debug)
