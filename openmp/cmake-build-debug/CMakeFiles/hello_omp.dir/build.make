# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /snap/clion/137/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/137/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cyberogg/hpc_course/hpc-20-21/openmp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/hello_omp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hello_omp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hello_omp.dir/flags.make

CMakeFiles/hello_omp.dir/hello.cpp.o: CMakeFiles/hello_omp.dir/flags.make
CMakeFiles/hello_omp.dir/hello.cpp.o: ../hello.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hello_omp.dir/hello.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hello_omp.dir/hello.cpp.o -c /home/cyberogg/hpc_course/hpc-20-21/openmp/hello.cpp

CMakeFiles/hello_omp.dir/hello.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hello_omp.dir/hello.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cyberogg/hpc_course/hpc-20-21/openmp/hello.cpp > CMakeFiles/hello_omp.dir/hello.cpp.i

CMakeFiles/hello_omp.dir/hello.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hello_omp.dir/hello.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cyberogg/hpc_course/hpc-20-21/openmp/hello.cpp -o CMakeFiles/hello_omp.dir/hello.cpp.s

# Object files for target hello_omp
hello_omp_OBJECTS = \
"CMakeFiles/hello_omp.dir/hello.cpp.o"

# External object files for target hello_omp
hello_omp_EXTERNAL_OBJECTS =

hello_omp: CMakeFiles/hello_omp.dir/hello.cpp.o
hello_omp: CMakeFiles/hello_omp.dir/build.make
hello_omp: CMakeFiles/hello_omp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hello_omp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hello_omp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hello_omp.dir/build: hello_omp

.PHONY : CMakeFiles/hello_omp.dir/build

CMakeFiles/hello_omp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hello_omp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hello_omp.dir/clean

CMakeFiles/hello_omp.dir/depend:
	cd /home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cyberogg/hpc_course/hpc-20-21/openmp /home/cyberogg/hpc_course/hpc-20-21/openmp /home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug /home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug /home/cyberogg/hpc_course/hpc-20-21/openmp/cmake-build-debug/CMakeFiles/hello_omp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hello_omp.dir/depend

