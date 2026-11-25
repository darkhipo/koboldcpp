// DYLD interposer recording every posix_madvise()/madvise() call of the process it is injected
// into:  clang -dynamiclib -o libmadvise_trace.dylib tools/madvise_trace.c
//         DYLD_INSERT_LIBRARIES=$PWD/libmadvise_trace.dylib MADVISE_TRACE_FILE=t.txt ./llamaserver ...
// Each call appends "<symbol> pid=<pid> advice=<n> len=<bytes>" and is forwarded unchanged.
// POSIX_MADV_WILLNEED == 3 on macOS; llama.cpp's loader issues it via posix_madvise() only.
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

typedef int (* advice_fn)(void * addr, size_t len, int advice);

static advice_fn real_posix_madvise = NULL, real_madvise = NULL;
static const char * trace_path = NULL;

static void trace(const char * sym, size_t len, int advice) {
    if (trace_path == NULL) {
        return;
    }
    const int fd = open(trace_path, O_WRONLY | O_APPEND | O_CREAT, 0644);  // O_APPEND keeps lines whole
    if (fd < 0) {
        return;
    }
    char line[128];
    const int n = snprintf(line, sizeof(line), "%s pid=%d advice=%d len=%zu\n", sym, (int) getpid(), advice, len);
    if (n > 0) { (void) write(fd, line, (size_t) n); }
    close(fd);
}

int posix_madvise(void * addr, size_t len, int advice) {
    trace("posix_madvise", len, advice);
    return real_posix_madvise ? real_posix_madvise(addr, len, advice) : EINVAL;
}

int madvise(void * addr, size_t len, int advice) {
    trace("madvise", len, advice);
    return real_madvise ? real_madvise(addr, len, advice) : EINVAL;
}

// Resolve real functions and the trace path before main(): dlsym()/getenv() may allocate or lock.
__attribute__((constructor)) static void madvise_trace_init(void) {
    real_posix_madvise = (advice_fn) dlsym(RTLD_NEXT, "posix_madvise");
    real_madvise       = (advice_fn) dlsym(RTLD_NEXT, "madvise");
    trace_path         = getenv("MADVISE_TRACE_FILE");
}
