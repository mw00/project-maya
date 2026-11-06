"""Compile the production reader loop with fake I/O; skipped predictions must not lap unfinished reads."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ReaderLifetimeTest(unittest.TestCase):
    def test_skipped_predictions_preserve_unfinished_readers(self):
        root = Path(__file__).resolve().parents[2]
        source = Path(os.environ.get('MAYA_PREFILL_READER_SOURCE', root / 'src/core/glm_prefill.cu')).read_text()
        start = source.index('    const auto reader_one = [&] {')
        loop = source[start:source.index('    std::thread reader([&]', start)]
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        harness = r'''
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include <cstdint>
int cudaSetDevice(int) { return 0; }
int cudaEventSynchronize(int) { return 0; }
int main() {
    constexpr int KL=2, dev_=0;
    struct Queue {
        std::mutex mu;
        std::vector<std::pair<int,int>> q{{0,0},{0,1},{0,2},{0,3}};
        std::atomic<int> n{4}, landed{0}, copied{4};
        std::atomic<bool> stop{false};
    } dq; // consumer discarded the first four predictions before I/O finished
    uint8_t bytes[KL]{};
    struct State { uint8_t* gpin; size_t gstride; std::vector<int> ev_land; } state{bytes,1,{0,0}};
    auto* S=&state;
    std::atomic<int> rnext{0}, entered{0};
    std::atomic<bool> live[KL]{};
    std::atomic<bool> overlap{false};
    std::mutex land_mu, io_mu;
    std::condition_variable io_cv;
    bool release_first=false;
    std::vector<int> land_seq(KL,-1);
    const auto helper_read = [&](int,int e,uint8_t*) {
        if (live[e%KL].exchange(true)) overlap=true;
        entered.fetch_or(1<<e);
        if (e==0) {
            std::unique_lock<std::mutex> lock(io_mu);
            io_cv.wait(lock,[&]{return release_first;});
        }
        live[e%KL]=false;
        return true;
    };
    const auto fast_read_part=[](int,int,int,uint8_t*,int,int) {};
''' + loop + r'''
    std::thread a(reader_one), b(reader_one);
    const auto wait_for = [&](auto pred) {
        const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(250);
        while (!pred() && std::chrono::steady_clock::now()<end) std::this_thread::yield();
        return pred();
    };
    // The second reader has claimed the next generation of the first slot.
    const bool attempted=wait_for([&]{return rnext.load()>=3;});
    // Old code enters all four reads and overlaps the blocked first one.
    // Correct code must wait until that first generation has been published.
    wait_for([&]{return entered.load()==15;});
    const bool reused_early=overlap.load() || (entered.load() & 12);
    { std::lock_guard<std::mutex> lock(io_mu); release_first=true; }
    io_cv.notify_all();
    const bool finished=wait_for([&]{return dq.landed.load()==4;});
    dq.stop=true; a.join(); b.join();
    if (!attempted || reused_early || !finished || overlap.load()) {
        std::fprintf(stderr,"attempted=%d reused_early=%d finished=%d overlap=%d landed=%d\n",
                     attempted,reused_early,finished,overlap.load(),dq.landed.load());
        return 1;
    }
    std::puts("PASS: skipped predictions cannot overwrite unpublished read generations");
}
'''
        with tempfile.TemporaryDirectory() as d:
            cpp = Path(d) / 'reader.cpp'
            binary = Path(d) / 'reader'
            cpp.write_text(harness)
            subprocess.run([compiler, '-std=c++17', '-O2', '-DNDEBUG', '-pthread', str(cpp), '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
