# compile:
#	g++-10 -fopenmp -std=c++11  -O3 -march=native  -ffast-math -o Caustics3D main.cpp
#	./Caustics3D

build_dir := ./build
cflags := -O3 -std=c++20
libs :=

.PHONY: clean build test

build: $(build_dir)/caustics3d

clean:
	-rm -r $(build_dir)

test: $(build_dir)/run-tests
	$(build_dir)/run-tests

############################################################
# HDF5 Support
############################################################
# Get HDF5 compilation flags using pkg-config
# hdf5_cflags = $(shell pkg-config --cflags hdf5)
# hdf5_libs = $(shell pkg-config --libs hdf5) -lhdf5_cpp

# Get HDF5 compilation flags manually if pkg-config does not work
hdf5_cc := $(shell which h5c++)
hdf5_root := $(shell dirname `dirname $(hdf5_cc)`)
hdf5_cflags := $(shell h5c++ -show '%' | cut -d% -f1 | cut -d' ' -f2-) -I$(hdf5_root)/include
hdf5_libs := $(shell h5c++ -show '%' | cut -d% -f2 | cut -d' ' -f2-)

############################################################
# 3rd party libraries
############################################################
argagg_cflags := -Iext/argagg/include
fmt_cflags := -Iext/fmt/include -DFMT_HEADER_ONLY
eigen_cflags := $(shell pkg-config --cflags eigen3)

############################################################
# Generic commands and arguments
############################################################
compile = g++
compile_flags = -iquote src -iquote include $(cflags) $(hdf5_cflags) $(argagg_cflags) $(fmt_cflags) $(eigen_cflags)
link = g++
link_flags = -lstdc++fs -lfmt $(libs) $(hdf5_libs)

############################################################
# Find sources, target object files and dep files
############################################################
cc_files := $(shell find src -name *.cpp -and -not -path src/main.cpp)
obj_files := $(cc_files:%.cpp=$(build_dir)/%.o)

main_obj_file := $(build_dir)/src/main.o

test_sources := $(shell find test -name *.cpp)
test_obj_files := $(test_sources:%.cc=$(build_dir)/%.o)

dep_files := $(obj_files:%.o=%.d) $(build_dir)/src/main.d $(test_obj_files:%.o=%.d)

############################################################
# Rules
############################################################
-include $(dep_files)

$(build_dir)/%.o: %.cpp
	@echo [Compiling]: $<
	@mkdir -p $(@D)
	$(compile) $(compile_flags) -MMD -c $< -o $@

# Link main executable
$(build_dir)/caustics3d: $(obj_files) $(main_obj_file)
	@echo [Linking] $@
	@mkdir -p $(@D)
	@$(link) $^ $(link_flags) -o $@

# Link testing exectuable
$(build_dir)/run-tests: $(obj_files) $(test_obj_files)
	@echo [Linking] $@: $^
	@mkdir -p $(@D)
	@$(link) $^ $(link_flags) -lgtest -lgmock -lpthread -o $@

############################################################
# Python module
############################################################
# python_sources := $(shell find python/src -name *.cc)
# python_build_dir := $(build_dir)/python
# python_cflags := -fPIC -fvisibility=hidden $(shell python -m pybind11 --includes)
# # python_libs = $(shell pkg-config --libs python3)
# python_obj_files := $(python_sources:python/%.cc=$(python_build_dir)/%.o)
# python_dep_files := $(python_sources:python/%.cc=$(python_build_dir)/%.d)
# python_target := $(python_build_dir)/bvh/_internal$(shell python3-config --extension-suffix)
# python_pkg_files := pyproject.toml setup.cfg bvh test
# 
# -include $(python_dep_files)
# 
# python-module: $(python_pkg_files:%=$(python_build_dir)/%) $(python_target)
# 
# $(python_build_dir)/%.o: python/%.cc
# 	@echo "[Compiling] $@"
# 	@mkdir -p $(@D)
# 	@$(compile) $(compile_flags) $(python_cflags) -fPIC -MMD -c $< -o $@
# 
# $(python_pkg_files:%=$(python_build_dir)/%): $(python_build_dir)/%: python/%
# 	@echo "[Copying]   $<"
# 	@mkdir -p $(@D)
# 	@cp -Tr $^ $@
# 
# $(python_target): $(python_obj_files)
# 	@echo "[Linking]   $@"
# 	@mkdir -p $(@D)
# 	@$(link) $^ $(link_flags) -shared -o $@
# 
