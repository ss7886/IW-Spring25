import os

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

def compile_library(output_path: str, lib_path: str, header_path: str):
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
    # ffi.set_source(output_path, f'#include "{header_path}"', libraries=[lib_path], library_directory=".")
    ffi.set_source(output_path, f'#include "{header_path}"', sources=["tree.c"], extra_compile_args=['-U', 'NDEBUG'])
    ffi.compile()

if __name__ == "__main__":
    lib = compile_library("_tree_cffi", "tree", "tree.h")
