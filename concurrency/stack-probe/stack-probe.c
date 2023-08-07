#include "thread.h"

void * volatile low[64];
void * volatile high[64];

void update_range(int T, void *ptr) {
    if (ptr < low[T]) low[T] = ptr;
    if (ptr > high[T]) high[T] = ptr;
}

void probe(int T, int n) {
  // 栈上变量连续分配
  // n 变量是每个栈上分配的，可以认为它就是栈的起始地址
  update_range(T, &n);
  long sz = (uintptr_t)high[T] - (uintptr_t)low[T];
  // 每个 KB 上会分配多个调用栈【当前一个调用栈的大小是 48 B】 
  // sz % 1024 < 48 : 按每个 KB 打印
  // printf("Stack(T%d) mod = %ld \n", T,sz % 1024);
  if (sz % 1024 < 48) {
    printf("Stack(T%d) >= %ld KB\n", T, sz / 1024);
  }
  probe(T, n + 1);  // Infinite recursion
}

void Tprobe(int T) {
  low[T] = (void *)-1;
  high[T] = (void *)0;
  update_range(T, &T);
  probe(T, 0);
}

int main() {
  setbuf(stdout, NULL); // 关掉缓冲区，确保每个输出都可以打印
  for (int i = 0; i < 4; i++) {
    create(Tprobe);
  }
}