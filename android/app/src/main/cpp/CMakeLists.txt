cmake_minimum_required(VERSION 3.22)
project(kokoro_native C CXX)

# espeak-ng (GPL-3.0) is compiled straight from the pinned submodule; only the
# phonemizer path is needed, so audio back-ends (klatt, mbrola, sonic, async) are off.
set(ESPEAK ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../third_party/espeak-ng)
set(LIBESPEAK ${ESPEAK}/src/libespeak-ng)

add_library(ucd STATIC
  ${ESPEAK}/src/ucd-tools/src/case.c
  ${ESPEAK}/src/ucd-tools/src/categories.c
  ${ESPEAK}/src/ucd-tools/src/ctype.c
  ${ESPEAK}/src/ucd-tools/src/proplist.c
  ${ESPEAK}/src/ucd-tools/src/scripts.c
  ${ESPEAK}/src/ucd-tools/src/tostring.c)
target_include_directories(ucd PUBLIC ${ESPEAK}/src/ucd-tools/src/include)
set_property(TARGET ucd PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(espeak-ng STATIC
  ${LIBESPEAK}/common.c ${LIBESPEAK}/mnemonics.c ${LIBESPEAK}/error.c ${LIBESPEAK}/ieee80.c
  ${LIBESPEAK}/compiledata.c ${LIBESPEAK}/compiledict.c ${LIBESPEAK}/dictionary.c
  ${LIBESPEAK}/encoding.c ${LIBESPEAK}/intonation.c ${LIBESPEAK}/langopts.c
  ${LIBESPEAK}/numbers.c ${LIBESPEAK}/phoneme.c ${LIBESPEAK}/phonemelist.c
  ${LIBESPEAK}/readclause.c ${LIBESPEAK}/setlengths.c ${LIBESPEAK}/soundicon.c
  ${LIBESPEAK}/spect.c ${LIBESPEAK}/ssml.c ${LIBESPEAK}/synthdata.c
  ${LIBESPEAK}/synthesize.c ${LIBESPEAK}/tr_languages.c ${LIBESPEAK}/translate.c
  ${LIBESPEAK}/translateword.c ${LIBESPEAK}/voices.c ${LIBESPEAK}/wavegen.c
  ${LIBESPEAK}/speech.c ${LIBESPEAK}/espeak_api.c)
target_include_directories(espeak-ng PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/espeak-config ${ESPEAK}/src/include ${ESPEAK}/src/include/compat)
target_include_directories(espeak-ng PUBLIC ${ESPEAK}/src/include)
target_compile_definitions(espeak-ng PRIVATE LIBESPEAK_NG_EXPORT=1 PATH_ESPEAK_DATA="/nonexistent")
target_compile_definitions(espeak-ng INTERFACE LIBESPEAK_NG_EXPORT=1)
target_compile_options(espeak-ng PRIVATE -fPIC -fwrapv -w)
target_link_libraries(espeak-ng PRIVATE ucd)

add_library(kokoro_native SHARED espeak_jni.cpp xnnpack_jni.cpp)
target_link_libraries(kokoro_native espeak-ng ucd log dl)
