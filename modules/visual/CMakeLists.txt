# ---------------------------------------
#  CMake file for visualization module.
# ---------------------------------------

if(WINRT OR NOT HAVE_GLFW)
  ocv_module_disable(opencv_visual)
endif()

set(the_description "Visualization")
ocv_define_module(visual
    opencv_3d
    WRAP
      python
)

ocv_target_link_libraries(${the_module} LINK_PRIVATE glfw)
