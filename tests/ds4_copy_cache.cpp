// CPU-only Linux checks for the optional expert-copy I/O helpers; no model, GPU or global cache writes.
#include "../src/ds4/ds4_os.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sys/vfs.h>

static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
#if defined(_WIN32)
    return 0;
#else
    using namespace strata::ds4::os;
    std::string pattern=std::string(argc>1 ? argv[1] : "/tmp")+"/maya-copy-cache-XXXXXX";
    std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
    int fd=mkstemp(name.data());check(fd>=0,"create fixture");unlink(name.data());
    const size_t p=(size_t)sysconf(_SC_PAGESIZE), n=p*4096;
    std::vector<uint8_t> data(n),copy(n);
    for(size_t i=0;i<n;++i)data[i]=(uint8_t)((i*37+i/503)%251);
    check(write(fd,data.data(),n)==(ssize_t)n,"write fixture");check(fsync(fd)==0,"sync fixture");
    check(pread_par(fd,copy.data(),n/2,0),"first NUMA half");
    check(pread_par(fd,copy.data()+n/2,n/2,n/2),"second NUMA half");
    void* map=mmap(nullptr,n,PROT_READ,MAP_SHARED,fd,0);check(map!=MAP_FAILED,"map for mincore only");
    std::vector<uint8_t> before(n/p),after(n/p);check(mincore(map,n,before.data())==0,"before mincore");
    const uint64_t hinted=release_copied_cache(fd,1,n-2);
    check(hinted==n-2*p,"only inward-aligned whole pages hinted");
    check(mincore(map,n,after.data())==0,"after mincore");
    // DONTNEED is advisory: a busy filesystem or larger boundary folios can retain pages.
    // Residency is diagnostic, not a portability requirement of this byte-correctness test.
    size_t resident_before=0,resident_after=0;
    for(size_t i=1;i+1<after.size();++i) { resident_before+=before[i]&1;resident_after+=after[i]&1; }
    check(!(before.front()&1)||(after.front()&1),"leading partial page retained");
    check(!(before.back()&1)||(after.back()&1),"trailing partial page retained");
    check(copy==data,"independent RAM copies unchanged after hint");
    check(release_copied_cache(fd,1,p-2)==0,"partial-page range ignored");
    check(release_copied_cache(fd,0,0)==0,"zero range ignored");
    check(release_copied_cache(kBadFd,0,p)==0,"invalid fd rejected");
    check(release_copied_cache(fd,UINT64_MAX-1,16)==0,"overflow rejected");
    std::vector<uint8_t> staging,direct(n-2,0xFF);
    check(!copy_direct(kBadFd,1,n-2,direct.data(),staging),"missing direct fd requests fallback");
    check(pread_par(fd,direct.data(),n-2,1),"buffered fallback succeeds");
    check(std::equal(direct.begin(),direct.end(),data.begin()+1),"fallback bytes correct");
    const auto dfd=open_direct("/proc/self/fd/"+std::to_string(fd));
    bool direct_ok=dfd!=kBadFd && copy_direct(dfd,1,n-2,direct.data(),staging);
    if(direct_ok) {
        check(std::equal(direct.begin(),direct.end(),data.begin()+1),"unaligned direct bytes correct");
        check(copy_direct(dfd,n-p+1,p-1,direct.data(),staging),"unaligned range ending at EOF");
        check(std::equal(direct.begin(),direct.begin()+p-1,data.end()-p+1),"EOF bytes correct");
        check(!copy_direct(dfd,n,4096,direct.data(),staging),"past EOF fails for fallback");
        check(!copy_direct(dfd,UINT64_MAX-1,16,direct.data(),staging),"direct overflow rejected");
        check(!copy_direct(dfd,0,0,direct.data(),staging),"direct zero request rejected");
        // The real NVMe fixture separately measures eviction and bypass. tmpfs/CI may ignore O_DIRECT.
    }
    close_fd(dfd);close(fd);check(release_copied_cache(fd,0,p)==0,"closed fd failure handled");munmap(map,n);
    std::cout<<"PASS: bytes, two halves, range bounds, fallback, direct="<<(direct_ok?"tested":"unavailable")
             <<"; cached interior "<<resident_before<<" -> "<<resident_after<<" (advisory)\n";
#endif
}
