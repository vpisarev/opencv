# Seeds for the fast compiler probes (see OpenCVCompilerProbe.cmake).
#
# OPENCV_PROBE_KNOWN_FLAGS: options OpenCV (and opencv_contrib) may ask the compiler
# about through ocv_check_flag_support()/add_extra_compiler_option()/ocv_warnings_disable().
# They are probed together, one -fsyntax-only run per language (plus a re-run when the
# driver stops at an unknown option), before the first individual check; the individual
# checks then read the batched result. Options missing from this list are still handled,
# only one compiler run each. Options that need code generation (-m*, -O*), linking
# (-pthread, -flto, ...), -Werror, or a specific source file are never batched.
# Hooks (e.g. POST_DETECT_COMPILER) may list(APPEND OPENCV_PROBE_KNOWN_FLAGS ...).

set(OPENCV_PROBE_KNOWN_FLAGS
  # cmake/OpenCVCompilerOptions.cmake
  -fsigned-char
  -W -Wall -Wreturn-type -Wnon-virtual-dtor -Waddress -Wsequence-point
  -Wformat -Wformat-security
  -Wmissing-declarations -Wmissing-prototypes -Wstrict-prototypes -Wundef -Winit-self
  -Wpointer-arith -Wshadow -Wsign-promo -Wuninitialized -Wsuggest-override
  -Winconsistent-missing-override -Wimplicit-fallthrough=3
  -Wno-delete-non-virtual-dtor -Wno-unnamed-type-template-args -Wno-comment
  -Wno-deprecated-enum-enum-conversion -Wno-deprecated-anon-enum-enum-conversion
  -Wno-psabi -Wno-strict-overflow -Wno-semicolon-before-method-body
  -Wcast-align -Wstrict-aliasing=2
  -fdiagnostics-show-option -Qunused-arguments
  -ffunction-sections -fdata-sections
  -fvisibility=hidden -fvisibility-inlines-hidden
  -ffast-math -fno-fast-math -fno-finite-math-only
  -fomit-frame-pointer -fno-omit-frame-pointer -fno-tree-vectorize
  # modules/highgui, modules/core
  -Wcast-function-type-strict
  # ocv_warnings_disable() call sites in opencv and opencv_contrib
  -Wno-undef -Wno-missing-declarations -Wno-shadow -Wno-unused-parameter
  -Wno-missing-prototypes -Wno-sign-compare -Wno-implicit-fallthrough
  -Wno-unused-but-set-variable -Wno-strict-prototypes -Wno-suggest-override
  -Wno-unused-function -Wno-strict-aliasing -Wno-deprecated-declarations
  -Wno-inconsistent-missing-override -Wno-shorten-64-to-32 -Wno-unused-variable
  -Wno-class-memaccess -Wno-ignored-qualifiers -Wno-maybe-uninitialized
  -Wno-deprecated-copy -Wno-enum-compare -Wno-unused -Wno-cast-align
  -Wno-cast-function-type -Wno-stringop-overflow -Wno-extra-semi
  -Wno-implicit-function-declaration -Wno-uninitialized -Wno-switch -Wno-deprecated
  -Wno-sign-promo -Wno-overloaded-virtual -Wno-invalid-offsetof
  -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast -Wno-misleading-indentation
  -Wno-array-parameter -Wno-null-pointer-subtraction
  -Wno-implicit-const-int-float-conversion -Wno-documentation
  -Wno-expansion-to-defined -Wno-non-virtual-dtor -Wno-attributes
  -Wno-shift-negative-value -Wno-unused-but-set-parameter -Wno-pointer-compare
  -Wno-absolute-value -Wno-parentheses -Wno-array-bounds -Wno-stringop-truncation
  -Wno-tautological-negation-compare -Wno-unused-local-typedefs
  -Wno-tautological-undefined-compare -Wno-extra -Wno-unused-const-variable
  -Wno-enum-compare-switch -Wno-stringop-overread -Wno-comma -Wno-type-limits
  -Wno-unused-private-field -Wno-tautological-compare
)

# ----------------------------------------------------------------------------
# System checks (headers, symbols, type sizes, source snippets) that OpenCV and its
# bundled 3rdparty libraries perform through CMake's check_*() modules. All of them
# are answered by compiling one generated C file (ocv_probe_seed_system_checks()),
# and the results are stored under the exact variable names and values the check_*()
# modules would produce, so the modules skip their own try_compile() ("if(NOT DEFINED ...)").
#
# Entry format: "KIND|VAR|ARG|HEADERS|LIBS|REQUIRES"  ("-" = empty field)
#   HDR    check_include_file(ARG VAR)
#   HDRS   check_include_files(ARG VAR)
#   SYM    check_symbol_exists(ARG HEADERS VAR)        HEADERS: comma separated, mandatory
#   FUNC   check_function_exists(ARG VAR)              HEADERS: where ARG is declared (any that exists)
#   TYPE   check_type_size(ARG VAR)                    HEADERS: CMAKE_EXTRA_INCLUDE_FILES
#   SRC    check_c_source_compiles(code VAR)           code in OPENCV_PROBE_SRC_<VAR>, -D in OPENCV_PROBE_SRC_DEFS_<VAR>
#   SRCPOS as SRC, but only a positive result is recorded (the consumer tries alternatives on failure)
#   LIBS: CMAKE_REQUIRED_LIBRARIES of the original check (used by the verification mode only)
#   REQUIRES: if() condition evaluated after the preceding entries were recorded
#
# Semantic differences to the CMake modules (documented, covered by OPENCV_VERIFY_FAST_PROBES):
#   check_include_file() compiles the header, HDR only tests that it exists (__has_include);
#   check_function_exists() links an undeclared symbol, FUNC tests for a declaration in HEADERS.
function(ocv_probe_collect_system_seeds outvar)
  set(__s "")
  # --- CMakeLists.txt (root)
  if(UNIX OR MINGW)
    if(NOT APPLE)
      list(APPEND __s "HDR|HAVE_PTHREAD|pthread.h|-|-|-")
    endif()
    if(OPENCV_ENABLE_MEMALIGN)
      list(APPEND __s "SYM|HAVE_POSIX_MEMALIGN|posix_memalign|stdlib.h|-|-"
                      "HDR|HAVE_MALLOC_H|malloc.h|-|-|-"
                      "SYM|HAVE_MEMALIGN|memalign|malloc.h|-|HAVE_MALLOC_H")
    endif()
    list(APPEND __s "SYM|HAVE_GETAUXVAL|getauxval|sys/auxv.h|-|-"
                    "SYM|HAVE_ELF_AUX_INFO|elf_aux_info|sys/auxv.h|-|-")
  endif()
  # --- 3rdparty/zlib
  if(NOT WITH_ZLIB_NG AND BUILD_ZLIB)
    list(APPEND __s "FUNC|HAVE_FSEEKO|fseeko|stdio.h|-|-")
    if(NOT MSVC)
      list(APPEND __s "HDR|HAVE_UNISTD_H|unistd.h|-|-|-")
    endif()
    list(APPEND __s "HDR|HAVE_STDARG_H|stdarg.h|-|-|-"
                    "TYPE|OFF64_T|off64_t|-|-|-")
  endif()
  # --- 3rdparty/libjpeg-turbo
  if(WITH_JPEG AND BUILD_JPEG)
    list(APPEND __s "TYPE|SIZE_T|size_t|-|-|-"
                    "TYPE|UNSIGNED_LONG|unsigned long|-|-|-"
                    "SRC|HAVE_BUILTIN_CTZL|-|-|-|SIZEOF_SIZE_T EQUAL SIZEOF_UNSIGNED_LONG")
    set(OPENCV_PROBE_SRC_HAVE_BUILTIN_CTZL "int main(int argc, char **argv) { unsigned long a = argc;  return __builtin_ctzl(a); }" PARENT_SCOPE)
    if(UNIX)
      list(APPEND __s "HDRS|HAVE_LOCALE_H|locale.h|-|-|-"
                      "HDRS|HAVE_STDDEF_H|stddef.h|-|-|-"
                      "HDRS|HAVE_STDLIB_H|stdlib.h|-|-|-"
                      "HDRS|NEED_SYS_TYPES_H|sys/types.h|-|-|-")
    endif()
    # simd/CMakeLists.txt, AArch64 only (32-bit ARM checks need CMAKE_REQUIRED_FLAGS)
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" __proc)
    list(LENGTH CMAKE_OSX_ARCHITECTURES __narch)
    if(__narch EQUAL 1 AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
      set(__aarch64 1)
    elseif(__narch EQUAL 0 AND CMAKE_SIZEOF_VOID_P EQUAL 8 AND (__proc STREQUAL "aarch64" OR __proc MATCHES "^arm"))
      set(__aarch64 1)
    else()
      set(__aarch64 0)
    endif()
    if(__aarch64 AND ((NOT DEFINED ENABLE_LIBJPEG_TURBO_SIMD AND NOT CV_DISABLE_OPTIMIZATION) OR ENABLE_LIBJPEG_TURBO_SIMD))
      list(APPEND __s "SRC|HAVE_VLD1_S16_X3|-|-|-|-"
                      "SRC|HAVE_VLD1_U16_X2|-|-|-|-"
                      "SRC|HAVE_VLD1Q_U8_X4|-|-|-|-")
      set(OPENCV_PROBE_SRC_HAVE_VLD1_S16_X3 "
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    int16_t input[12];
    int16x4x3_t output;
    int i;
    for (i = 0; i < 12; i++) input[i] = (int16_t)argc;
    output = vld1_s16_x3(input);
    vst3_s16(input, output);
    return (int)input[0];
  }" PARENT_SCOPE)
      set(OPENCV_PROBE_SRC_HAVE_VLD1_U16_X2 "
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    uint16_t input[8];
    uint16x4x2_t output;
    int i;
    for (i = 0; i < 8; i++) input[i] = (uint16_t)argc;
    output = vld1_u16_x2(input);
    vst2_u16(input, output);
    return (int)input[0];
  }" PARENT_SCOPE)
      set(OPENCV_PROBE_SRC_HAVE_VLD1Q_U8_X4 "
  #include <arm_neon.h>
  int main(int argc, char **argv) {
    uint8_t input[64];
    uint8x16x4_t output;
    int i;
    for (i = 0; i < 64; i++) input[i] = (uint8_t)argc;
    output = vld1q_u8_x4(input);
    vst4q_u8(input, output);
    return (int)input[0];
  }" PARENT_SCOPE)
    endif()
  endif()
  # --- 3rdparty/libtiff
  if(WITH_TIFF AND BUILD_TIFF)
    list(APPEND __s "HDR|HAVE_ASSERT_H|assert.h|-|-|-")
    if(NOT MSVC)
      list(APPEND __s "HDR|HAVE_DLFCN_H|dlfcn.h|-|-|-")
    endif()
    list(APPEND __s "HDR|HAVE_FCNTL_H|fcntl.h|-|-|-"
                    "HDR|HAVE_INTTYPES_H|inttypes.h|-|-|-"
                    "HDR|HAVE_IO_H|io.h|-|-|-"
                    "HDR|HAVE_LIMITS_H|limits.h|-|-|-"
                    "HDR|HAVE_MALLOC_H|malloc.h|-|-|-"
                    "HDR|HAVE_MEMORY_H|memory.h|-|-|-"
                    "HDR|HAVE_SEARCH_H|search.h|-|-|-"
                    "HDR|HAVE_STDINT_H|stdint.h|-|-|-"
                    "HDR|HAVE_STRING_H|string.h|-|-|-")
    if(NOT MSVC)
      list(APPEND __s "HDR|HAVE_STRINGS_H|strings.h|-|-|-"
                      "HDR|HAVE_SYS_TIME_H|sys/time.h|-|-|-")
    endif()
    list(APPEND __s "HDR|HAVE_SYS_TYPES_H|sys/types.h|-|-|-")
    if(NOT MSVC)
      list(APPEND __s "HDR|HAVE_UNISTD_H|unistd.h|-|-|-")
    endif()
    list(APPEND __s "SRCPOS|C_HAS_inline|-|-|-|-")
    set(OPENCV_PROBE_SRC_C_HAS_inline "
        typedef int foo_t;
        static inline foo_t static_foo() {return 0;}
        foo_t foo(){return 0;}
        int main(int argc, char *argv[]) {return 0;}" PARENT_SCOPE)
    set(OPENCV_PROBE_SRC_DEFS_C_HAS_inline "-Dinline=inline" PARENT_SCOPE)
    list(APPEND __s "TYPE|SIZEOF_SIGNED_SHORT|signed short|-|-|-"
                    "TYPE|SIZEOF_UNSIGNED_SHORT|unsigned short|-|-|-"
                    "TYPE|SIZEOF_SIGNED_INT|signed int|-|-|-"
                    "TYPE|SIZEOF_UNSIGNED_INT|unsigned int|-|-|-"
                    "TYPE|SIZEOF_SIGNED_LONG|signed long|-|-|-"
                    "TYPE|SIZEOF_UNSIGNED_LONG|unsigned long|-|-|-"
                    "TYPE|SIZEOF_SIGNED_LONG_LONG|signed long long|-|-|-"
                    "TYPE|SIZEOF_UNSIGNED_LONG_LONG|unsigned long long|-|-|-"
                    "TYPE|SIZEOF_UNSIGNED_CHAR_P|unsigned char *|-|-|-"
                    "TYPE|SIZEOF_SIZE_T|size_t|stddef.h|-|-"
                    "TYPE|SIZEOF_PTRDIFF_T|ptrdiff_t|stddef.h|-|-")
    if(NOT MSVC)
      list(APPEND __s "TYPE|int8|INT8|-|-|-"
                      "TYPE|int16|INT16|-|-|-"
                      "TYPE|int32|INT32|-|-|-"
                      "FUNC|HAVE_FLOOR|floor|math.h|m|-"
                      "FUNC|HAVE_POW|pow|math.h|m|-"
                      "FUNC|HAVE_SQRT|sqrt|math.h|m|-"
                      "FUNC|HAVE_ISASCII|isascii|ctype.h|-|-"
                      "FUNC|HAVE_MEMSET|memset|string.h|-|-"
                      "FUNC|HAVE_MMAP|mmap|sys/mman.h|-|-"
                      "FUNC|HAVE_GETOPT|getopt|unistd.h,getopt.h|-|-")
    endif()
    list(APPEND __s "FUNC|HAVE_MEMMOVE|memmove|string.h|-|-"
                    "FUNC|HAVE_SETMODE|setmode|io.h,unistd.h|-|-"
                    "FUNC|HAVE_STRCASECMP|strcasecmp|strings.h,string.h|-|-"
                    "FUNC|HAVE_STRCHR|strchr|string.h|-|-"
                    "FUNC|HAVE_STRRCHR|strrchr|string.h|-|-"
                    "FUNC|HAVE_STRSTR|strstr|string.h|-|-"
                    "FUNC|HAVE_STRTOL|strtol|stdlib.h|-|-"
                    "FUNC|HAVE_STRTOUL|strtol|stdlib.h|-|-"   # sic: libtiff checks strtol for HAVE_STRTOUL
                    "FUNC|HAVE_STRTOULL|strtoull|stdlib.h|-|-"
                    "FUNC|HAVE_LFIND|lfind|search.h|-|-")
  endif()
  # --- 3rdparty/openjpeg
  if(WITH_OPENJPEG AND BUILD_OPENJPEG)
    list(APPEND __s "HDR|HAVE_STRING_H|string.h|-|-|-"
                    "HDR|HAVE_MEMORY_H|memory.h|-|-|-"
                    "HDR|HAVE_STDLIB_H|stdlib.h|-|-|-"
                    "HDR|HAVE_STDIO_H|stdio.h|-|-|-"
                    "HDR|HAVE_MATH_H|math.h|-|-|-"
                    "HDR|HAVE_FLOAT_H|float.h|-|-|-"
                    "HDR|HAVE_TIME_H|time.h|-|-|-"
                    "HDR|HAVE_STDARG_H|stdarg.h|-|-|-"
                    "HDR|HAVE_CTYPE_H|ctype.h|-|-|-"
                    "HDR|HAVE_ASSERT_H|assert.h|-|-|-"
                    "HDR|OPJ_HAVE_STDINT_H|stdint.h|-|-|-"
                    "HDR|OPJ_HAVE_INTTYPES_H|inttypes.h|-|-|-"
                    "HDR|HAVE_STRINGS_H|strings.h|-|-|-"
                    "HDR|HAVE_SYS_STAT_H|sys/stat.h|-|-|-"
                    "HDR|HAVE_SYS_TYPES_H|sys/types.h|-|-|-"
                    "HDR|HAVE_UNISTD_H|unistd.h|-|-|-"
                    "HDRS|OPJ_HAVE_MALLOC_H|malloc.h|-|-|-"
                    "SYM|OPJ_HAVE__ALIGNED_MALLOC|_aligned_malloc|malloc.h|-|-"
                    "SYM|OPJ_HAVE_POSIX_MEMALIGN|posix_memalign|stdlib.h|-|-"
                    "SYM|OPJ_HAVE_MEMALIGN|memalign|malloc.h|-|-")
  endif()
  # --- modules/videoio/cmake/detect_v4l.cmake
  if(WITH_V4L AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND __s "HDR|HAVE_CAMV4L2|linux/videodev2.h|-|-|-"
                    "HDR|HAVE_VIDEOIO|sys/videoio.h|-|-|-")
  endif()
  # --- modules/videoio/cmake/detect_obsensor.cmake
  if(WITH_OBSENSOR AND UNIX AND NOT OBSENSOR_USE_ORBBEC_SDK)
    list(APPEND __s "HDR|HAVE_CAMV4L2_OBSENSOR|linux/videodev2.h|-|-|-")
  endif()
  # --- CMake's CheckTypeSize prelude
  list(APPEND __s "HDR|HAVE_SYS_TYPES_H|sys/types.h|-|-|-"
                  "HDR|HAVE_STDINT_H|stdint.h|-|-|-"
                  "HDR|HAVE_STDDEF_H|stddef.h|-|-|-")
  set(${outvar} "${__s}" PARENT_SCOPE)
endfunction()
