# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/ARTEM/CLionProjects/openmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/mandelbrot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mandelbrot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mandelbrot.dir/flags.make

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o: CMakeFiles/mandelbrot.dir/flags.make
CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o: ../mandelbrot.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o -c /mnt/c/Users/ARTEM/CLionProjects/openmp/mandelbrot.cpp

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mandelbrot.dir/mandelbrot.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/ARTEM/CLionProjects/openmp/mandelbrot.cpp > CMakeFiles/mandelbrot.dir/mandelbrot.cpp.i

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mandelbrot.dir/mandelbrot.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/ARTEM/CLionProjects/openmp/mandelbrot.cpp -o CMakeFiles/mandelbrot.dir/mandelbrot.cpp.s

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.requires:

.PHONY : CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.requires

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.provides: CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.requires
	$(MAKE) -f CMakeFiles/mandelbrot.dir/build.make CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.provides.build
.PHONY : CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.provides

CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.provides.build: CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o


# Object files for target mandelbrot
mandelbrot_OBJECTS = \
"CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o"

# External object files for target mandelbrot
mandelbrot_EXTERNAL_OBJECTS =

mandelbrot: CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o
mandelbrot: CMakeFiles/mandelbrot.dir/build.make
mandelbrot: CMakeFiles/mandelbrot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mandelbrot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mandelbrot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mandelbrot.dir/build: mandelbrot

.PHONY : CMakeFiles/mandelbrot.dir/build

CMakeFiles/mandelbrot.dir/requires: CMakeFiles/mandelbrot.dir/mandelbrot.cpp.o.requires

.PHONY : CMakeFiles/mandelbrot.dir/requires

CMakeFiles/mandelbrot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mandelbrot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mandelbrot.dir/clean

CMakeFiles/mandelbrot.dir/depend:
	cd /mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/ARTEM/CLionProjects/openmp /mnt/c/Users/ARTEM/CLionProjects/openmp /mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug /mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug /mnt/c/Users/ARTEM/CLionProjects/openmp/cmake-build-debug/CMakeFiles/mandelbrot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mandelbrot.dir/depend
