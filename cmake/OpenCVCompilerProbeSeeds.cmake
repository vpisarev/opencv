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
