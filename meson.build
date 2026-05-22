project(
    'phj-bench',
    'cpp',
    version: '0.1.0',
    license: 'Apache-2.0',
    default_options: [
        'cpp_std=c++23',
        'buildtype=release',
        'warning_level=3',
        'b_ndebug=if-release',
        'b_lto=false',
    ],
    meson_version: '>=1.0',
)

add_project_arguments(
    [
        '-Wpedantic',
        '-Wshadow',
        '-Wnon-virtual-dtor',
        '-Wold-style-cast',
        '-Wcast-align',
        '-Wunused',
        '-Woverloaded-virtual',
        '-Wconversion',
        '-Wsign-conversion',
        '-Wnull-dereference',
        '-Wdouble-promotion',
        '-Wformat=2',
        '-fno-omit-frame-pointer',
    ],
    language: 'cpp',
)

cpp = meson.get_compiler('cpp')
avx512_args = [
    '-mavx512f',
    '-mavx512dq',
    '-mavx512bw',
    '-mavx512vl',
]
if cpp.has_multi_arguments(avx512_args)
    add_project_arguments(avx512_args, language: 'cpp')
endif

threads_dep = dependency('threads')

subdir('src')
