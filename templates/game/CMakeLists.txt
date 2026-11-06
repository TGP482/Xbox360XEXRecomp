# Game Title (Xbox 360), statically recompiled

cmake_minimum_required(VERSION 3.25)
project(game LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# the SDK: sdk/ here, two folders up inside the SDK's templates, a checkout beside this repository, else downloaded
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sdk/cmake/XexRecomp.cmake)
    set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/sdk)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/XexRecomp.cmake)
    set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/../..)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Xbox360XEX-Recomp/cmake/XexRecomp.cmake)
    set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/../Xbox360XEX-Recomp)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Xbox360XEXRecomp/cmake/XexRecomp.cmake)
    set(SDK ${CMAKE_CURRENT_SOURCE_DIR}/../Xbox360XEXRecomp)
else()
    include(FetchContent)
    FetchContent_Declare(xex_sdk GIT_REPOSITORY https://github.com/TGP482/Xbox360XEX-Recomp.git GIT_TAG main)
    FetchContent_MakeAvailable(xex_sdk)
    set(SDK ${xex_sdk_SOURCE_DIR})
endif()
include(${SDK}/cmake/XexRecomp.cmake)

file(GLOB XEX_HOST_SOURCES CONFIGURE_DEPENDS src/*/*.cpp)
list(FILTER XEX_HOST_SOURCES EXCLUDE REGEX "/src/game/")

# partitions as xex_symbols wrote them into config/game.toml
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS config/game.toml)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config/game.toml)
    file(STRINGS config/game.toml GAME_PARTITIONS REGEX "^partitions *=")
    string(REGEX MATCH "[0-9]+" GAME_PARTITIONS "${GAME_PARTITIONS}")
endif()
# no config yet: build.bat writes it with xex_symbols, the change reconfigures
if(NOT GAME_PARTITIONS)
    set(GAME_PARTITIONS 8)
endif()

# target name = GAME_EXE in build.bat, xex = GAME_XEX
add_xex_image(Game default.xex game.toml ${GAME_PARTITIONS} game)
