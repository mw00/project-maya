/* Metal-only, no model load: exercise GGML's fixed free-block table near capacity.
 * Include the isolated generated ggml allocator to reach its static test helpers.
 * Link dead-stripping excludes unrelated backend functions in that compilation unit.
 */
#include <stdio.h>
#include <stdlib.h>
#include "ggml-alloc.c"
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    (void) file; (void) line; (void) fmt;
    abort();
}
#define VERIFY(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)
int main(void) {
    struct tallocr_chunk * chunk = calloc(1, sizeof(*chunk));
    VERIFY(chunk);
    for (int i = 0; i < 4095; ++i) {
        ggml_dyn_tallocr_insert_block(chunk, (size_t)i * 16, 8);
        VERIFY(chunk->n_free_blocks == i + 1);
    }
    for (int i = 1; i < 4095; ++i) {
        VERIFY(chunk->free_blocks[i - 1].offset < chunk->free_blocks[i].offset);
    }
    ggml_dyn_tallocr_remove_block(chunk, 100);
    VERIFY(chunk->n_free_blocks == 4094);
    VERIFY(chunk->free_blocks[100].offset == 101 * 16);
    free(chunk);
    puts("Metal GGML allocator: 4095 free blocks, ordered insert/remove OK");
    return 0;
}
