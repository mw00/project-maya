#include "strata/core/glm_helper_link.hpp"

#include <cstdio>
#include <mutex>
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core::hlink {
namespace {
std::mutex g_mu;
std::vector<Block> g_blocks;

#ifdef __linux__
void* map_fd(const char* name, size_t bytes, int& fd) {
    fd = memfd_create(name, 0);   // (no MFD_CLOEXEC: the helper program inherits it)
    if (fd < 0) return nullptr;
    if (ftruncate(fd, (off_t) bytes) != 0 || fallocate(fd, 0, 0, (off_t) bytes) != 0) {
        ::close(fd);
        fd = -1;
        return nullptr;
    }
    void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (p == MAP_FAILED) {
        ::close(fd);
        fd = -1;
        return nullptr;
    }
    madvise(p, bytes, MADV_HUGEPAGE);   // (honoured where shmem_enabled allows it)
    return p;
}
#endif
}  // namespace

void* shared_alloc(size_t bytes) {
#ifdef __linux__
    int fd = -1;
    void* p = map_fd("maya-glm-tier", bytes, fd);
    if (p == nullptr) return nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    g_blocks.push_back({p, bytes, fd});
    std::fprintf(stderr, "glm helper link: RAM tier block %.2f GB in shared memory (fd %d)\n", bytes / 1073741824.0, fd);
    return p;
#else
    (void) bytes;
    return nullptr;
#endif
}

const std::vector<Block>& blocks() { return g_blocks; }

Ctl* ctl_alloc(int& fd) {
#ifdef __linux__
    void* p = map_fd("maya-glm-link", sizeof(Ctl), fd);
    if (p == nullptr) return nullptr;
    std::memset(p, 0, sizeof(Ctl));
    Ctl* c = (Ctl*) p;
    c->magic = kMagic;
    return c;
#else
    fd = -1;
    return nullptr;
#endif
}

}  // namespace strata::core::hlink
