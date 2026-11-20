# Xbox 360 static recompilation: xex_recomp, the host runtime and add_xex_image()
include_guard(GLOBAL)

get_filename_component(XEX_SDK_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
# build scripts find the SDK's xdk folder and THIRD_PARTY.txt through this
file(TO_NATIVE_PATH "${XEX_SDK_DIR}" XEX_SDK_DIR_NATIVE)
file(WRITE ${CMAKE_BINARY_DIR}/sdk_dir.txt "${XEX_SDK_DIR_NATIVE}\n")

include(FetchContent)
FetchContent_Declare(tomlplusplus GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git GIT_TAG v3.4.0
    GIT_SHALLOW TRUE)
FetchContent_Declare(imgui GIT_REPOSITORY https://github.com/ocornut/imgui.git GIT_TAG v1.91.0 GIT_SHALLOW TRUE)
# SDL3 for controllers, static so it links into the exe
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_RENDER OFF CACHE BOOL "" FORCE)
set(SDL_GPU OFF CACHE BOOL "" FORCE)
set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
# guest threads read the pads, which SDL's debug-only main thread checks reject
set(SDL_ASSERTIONS release CACHE STRING "" FORCE)
FetchContent_Declare(SDL3 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git GIT_TAG release-3.4.16 GIT_SHALLOW TRUE)
# FFmpeg's LGPL shared build for XMA (libavcodec's XMA2 decoder): the runtime loads its DLLs, the headers match them
FetchContent_Declare(ffmpeg
    URL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip)
FetchContent_MakeAvailable(tomlplusplus imgui SDL3 ffmpeg)
file(GLOB XEX_FFMPEG_DLLS ${ffmpeg_SOURCE_DIR}/bin/avcodec-*.dll ${ffmpeg_SOURCE_DIR}/bin/avutil-*.dll
    ${ffmpeg_SOURCE_DIR}/bin/swresample-*.dll)

# What the game needs to build its code on the player's PC (runtime/app/game_code.cpp), beside the executables: the
# recompiler, clang and lld from the toolchain building this, the game's configs, the XDK layouts they include and the
# headers the generated code includes
set(XEX_COMPILER_DIR ${CMAKE_BINARY_DIR}/compiler)
get_filename_component(XEX_LLVM_BIN ${CMAKE_CXX_COMPILER} DIRECTORY)
file(GLOB XEX_GAME_CONFIGS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/config/*.toml)
set(XEX_COMPILER_HEADERS ${XEX_SDK_DIR}/runtime/cpu/ppc_ops.h ${XEX_SDK_DIR}/runtime/cpu/ppc_vmx.h
    ${XEX_SDK_DIR}/runtime/cpu/ppc_context.h)
file(GLOB XEX_FREESTANDING_HEADERS CONFIGURE_DEPENDS ${XEX_SDK_DIR}/runtime/cpu/freestanding/*)
file(GLOB XEX_LAYOUTS CONFIGURE_DEPENDS ${XEX_SDK_DIR}/xdk/*/layout.toml)
set(XEX_COPY_LAYOUTS)
foreach(layout ${XEX_LAYOUTS})
    get_filename_component(build_dir ${layout} DIRECTORY)
    get_filename_component(build ${build_dir} NAME)
    list(APPEND XEX_COPY_LAYOUTS COMMAND ${CMAKE_COMMAND} -E make_directory ${XEX_COMPILER_DIR}/xdk/${build}
         COMMAND ${CMAKE_COMMAND} -E copy_if_different ${layout} ${XEX_COMPILER_DIR}/xdk/${build})
endforeach()
add_custom_target(compiler_files ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${XEX_COMPILER_DIR}/config ${XEX_COMPILER_DIR}/include/cpu
            ${XEX_COMPILER_DIR}/include/freestanding
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_LLVM_BIN}/clang.exe ${XEX_LLVM_BIN}/lld-link.exe
            ${XEX_SDK_DIR}/tools/LLVM-LICENSE.txt ${XEX_COMPILER_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_GAME_CONFIGS} ${XEX_COMPILER_DIR}/config
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_COMPILER_HEADERS} ${XEX_COMPILER_DIR}/include/cpu
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_FREESTANDING_HEADERS} ${XEX_COMPILER_DIR}/include/freestanding
    ${XEX_COPY_LAYOUTS}
    VERBATIM)

# PowerPC to C++ recompiler, with the runtime's XEX loader
file(GLOB XEX_RECOMP_SOURCES CONFIGURE_DEPENDS ${XEX_SDK_DIR}/tools/recomp/*.cpp ${XEX_SDK_DIR}/runtime/xex/*.cpp)
add_executable(xex_recomp ${XEX_RECOMP_SOURCES})
set_target_properties(xex_recomp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${XEX_COMPILER_DIR})
target_include_directories(xex_recomp PRIVATE ${XEX_SDK_DIR}/runtime ${XEX_SDK_DIR}/tools/recomp)
target_compile_definitions(xex_recomp PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(xex_recomp PRIVATE tomlplusplus::tomlplusplus)

# XDK signature scanner that writes a new title's config
add_executable(xex_symbols ${XEX_SDK_DIR}/tools/symbols/main.cpp ${XEX_SDK_DIR}/runtime/xex/xex.cpp
    ${XEX_SDK_DIR}/runtime/xex/lzx.cpp)
target_include_directories(xex_symbols PRIVATE ${XEX_SDK_DIR}/runtime)
target_compile_definitions(xex_symbols PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(xex_symbols PRIVATE tomlplusplus::tomlplusplus)

set(XEX_IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_dx12.cpp)

file(GLOB XEX_RUNTIME_SOURCES CONFIGURE_DEPENDS ${XEX_SDK_DIR}/runtime/*/*.cpp)

# the shader compiler the renderer loads, dxcompiler.dll and dxil.dll: the newest Windows SDK's unless XEX_DXC_DIR says
if(NOT XEX_DXC_DIR)
    file(GLOB XEX_DXC_DIRS "C:/Program Files (x86)/Windows Kits/10/bin/10.*/x64")
    list(SORT XEX_DXC_DIRS COMPARE NATURAL ORDER DESCENDING)
    foreach(dir ${XEX_DXC_DIRS})
        if(EXISTS ${dir}/dxcompiler.dll AND EXISTS ${dir}/dxil.dll)
            set(XEX_DXC_DIR ${dir})
            break()
        endif()
    endforeach()
endif()
if(NOT XEX_DXC_DIR)
    message(WARNING "dxcompiler.dll and dxil.dll not found, set XEX_DXC_DIR to the folder holding them")
endif()
# SDL's headers go to the input code alone (it is linked LINK_ONLY), FFmpeg's to the XMA decoder
set_source_files_properties(${XEX_SDK_DIR}/runtime/core/input.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${sdl3_SOURCE_DIR}/include")
set_source_files_properties(${XEX_SDK_DIR}/runtime/audio/xma_decoder.cpp PROPERTIES
    INCLUDE_DIRECTORIES "${ffmpeg_SOURCE_DIR}/include")

# One executable per recompiled image: the SDK runtime, XEX_HOST_SOURCES shared by every image of the game, and
# src/<game_dir> hooks only this image has. The image's code goes to <target>.dll beside it, which the executable delay
# loads: built here from config/<config> (partitions as that config sets them, library = "<target>.dll"), while a
# release leaves it out and the executable builds it from the player's xex (runtime/app/game_code.cpp, which compiles
# with the same flags). Expects assets/<xex> and the config's out = "generated/<game_dir>". At run time the game may be
# a disc image instead.
function(add_xex_image target xex config partitions game_dir)
    set(GEN ${CMAKE_CURRENT_SOURCE_DIR}/generated/${game_dir})
    set(GEN_SOURCES ${GEN}/ppc_func_mapping.cpp)
    math(EXPR last "${partitions} - 1")
    foreach(i RANGE ${last})
        list(APPEND GEN_SOURCES ${GEN}/ppc_recomp.${i}.cpp)
    endforeach()
    file(GLOB configs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/config/*.toml ${XEX_SDK_DIR}/xdk/*/*.toml)
    add_custom_command(
        OUTPUT ${GEN}/codegen.stamp
        BYPRODUCTS ${GEN_SOURCES} ${GEN}/host_bodies.cpp ${GEN}/ppc_recomp_shared.h ${GEN}/image_symbols.h
                   ${GEN}/hook_exports.h ${GEN}/exports.def ${GEN}/imports.def ${GEN}/code.lib
        COMMAND xex_recomp config/${config} --xdk ${XEX_SDK_DIR}/xdk
        COMMAND ${XEX_LLVM_BIN}/lld-link.exe /lib /machine:x64 /def:${GEN}/imports.def /out:${GEN}/code.lib
        COMMAND ${CMAKE_COMMAND} -E touch ${GEN}/codegen.stamp
        DEPENDS xex_recomp ${configs} ${CMAKE_CURRENT_SOURCE_DIR}/assets/${xex}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Recompiling ${xex}"
        VERBATIM)
    add_custom_target(recompile_${target} DEPENDS ${GEN}/codegen.stamp)
    file(GLOB GAME_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${game_dir}/*.cpp)
    # the xex's SHA-1: the executable checks the game it finds is the release it was made from
    set(xex_file ${CMAKE_CURRENT_SOURCE_DIR}/assets/${xex})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${xex_file})
    file(SHA1 ${xex_file} xex_sha1)
    set(XEX_HASH ${CMAKE_CURRENT_BINARY_DIR}/${target}_xex.cpp)
    file(CONFIGURE OUTPUT ${XEX_HASH} CONTENT "extern const char kImageSha1[] = \"${xex_sha1}\";\n")
    # the recompiled bodies the executable calls, through the DLL's PPCBodies
    set_source_files_properties(${GEN}/host_bodies.cpp PROPERTIES GENERATED TRUE)
    add_executable(${target} WIN32 ${XEX_RUNTIME_SOURCES} ${XEX_HOST_SOURCES} ${GAME_SOURCES} ${XEX_IMGUI_SOURCES}
        ${XEX_HASH} ${GEN}/host_bodies.cpp)
    add_dependencies(${target} recompile_${target} compiler_files)
    # the import library of what the executable exports to its DLL goes to the compiler folder
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON ARCHIVE_OUTPUT_DIRECTORY ${XEX_COMPILER_DIR})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${XEX_SDK_DIR}/runtime ${GEN})
    target_include_directories(${target} SYSTEM PRIVATE ${imgui_SOURCE_DIR})
    target_compile_options(${target} PRIVATE -fno-strict-aliasing)
    target_compile_definitions(${target} PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries(${target} PRIVATE tomlplusplus::tomlplusplus $<LINK_ONLY:SDL3::SDL3-static> d3d12 dxgi dxguid
        d3dcompiler xaudio2 ole32 windowscodecs dbghelp winmm dwmapi ws2_32)
    target_link_libraries(${target} PRIVATE ${GEN}/code.lib delayimp)
    target_link_options(${target} PRIVATE -fuse-ld=lld -g -Wl,/delayload:${target}.dll)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_FFMPEG_DLLS} $<TARGET_FILE_DIR:${target}>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ffmpeg_SOURCE_DIR}/LICENSE.txt
                $<TARGET_FILE_DIR:${target}>/FFmpeg-LICENSE.txt)
    if(XEX_DXC_DIR)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${XEX_DXC_DIR}/dxcompiler.dll ${XEX_DXC_DIR}/dxil.dll
                    $<TARGET_FILE_DIR:${target}>)
    endif()

    # without a C runtime: the executable passes on what the code calls. Built for the CPU it runs on; the executable
    # stays at the x86-64-v2 floor. No fused multiply add or errno: PowerPC rounds each step, fma is explicit. No stack
    # probes: there is no __chkstk, guest threads run on committed host stacks. A higher inline threshold: a small
    # guest function's context loads and stores count against inlining, yet fold away once inlined.
    add_library(${target}_code SHARED ${GEN_SOURCES})
    set_target_properties(${target}_code PROPERTIES OUTPUT_NAME ${target} CXX_EXTENSIONS OFF)
    set_source_files_properties(${GEN_SOURCES} PROPERTIES GENERATED TRUE)
    add_dependencies(${target}_code recompile_${target})
    target_include_directories(${target}_code PRIVATE ${XEX_SDK_DIR}/runtime ${GEN})
    target_include_directories(${target}_code SYSTEM PRIVATE ${XEX_SDK_DIR}/runtime/cpu/freestanding)
    target_compile_definitions(${target}_code PRIVATE PPC_GAME_CODE)
    target_compile_options(${target}_code PRIVATE -fms-compatibility-version=19.40 -O2 -march=native -ffp-contract=off
        -fno-math-errno -mno-stack-arg-probe -fno-strict-aliasing -nostdinc -nostdinc++ -w
        "SHELL:-mllvm -inline-threshold=500")
    target_link_libraries(${target}_code PRIVATE ${target})
    target_link_options(${target}_code PRIVATE -fuse-ld=lld -Wl,/noentry,/nodefaultlib,/opt:noicf
        -Wl,/def:${GEN}/exports.def)
    # a stamp from an earlier build on the executable's first start would outrank this DLL
    add_custom_command(TARGET ${target}_code POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f ${XEX_COMPILER_DIR}/work/${target}/built.txt)
endfunction()
