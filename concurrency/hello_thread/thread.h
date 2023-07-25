#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>

#define NTHREAD 64
enum { T_FREE = 0, T_LIVE, T_DEAD, };
struct thread {
  int id, status;
  pthread_t thread;
  void (*entry)(int);
};

/*
tptr 是一个指向结构体数组的指针，用于指向线程池中的下一个空闲位置
在 create 函数中，它指向下一个空闲位置，然后将新线程的信息存储在该位置
在 join 函数中，它遍历线程池并等待所有线程完成执行
在 cleanup 函数中，它等待所有线程完成执行并释放线程池
*/
struct thread tpool[NTHREAD], *tptr = tpool;

void *wrapper(void *arg) {
  struct thread *thread = (struct thread *)arg;
  // 执行函数：entry 是传进来的 Thello 函数， thread->id 是 Thello 函数的参数
  thread->entry(thread->id);
  return NULL;
}

void create(void *fn) {
  /* 
  如果断言的条件为假（即为0），则会向stderr打印一条出错信息，并通过调用abort终止程序运行
  assert可以帮助程序员排查错误，但是也会影响程序的性能，所以在调试结束后，可以通过定义NDEBUG来禁用assert 
  */
  /*
  指针减法是地址差值/类型size
  栈上连续分配的int元素之间的差值是1
  指针加法加的是数据类型的size
  */
  assert(tptr - tpool < NTHREAD);
  *tptr = (struct thread) {
    // 线程池索引：0、1、2...
    .id = tptr - tpool + 1,
    .status = T_LIVE,
    .entry = fn,
  };
  /*
  &(tptr->thread)：用来保存创建出来的线程
  NULL：这个参数用来设置线程的属性，栈内存大小、调度策略...
  wrapper：创建出来的线程要执行的函数
  tptr：wrapper函数的参数
  */
  pthread_create(&(tptr->thread), NULL, wrapper, tptr);
  ++tptr;
}

void join() {
  for (int i = 0; i < NTHREAD; i++) {
    struct thread *t = &tpool[i];
    if (t->status == T_LIVE) {
      // 等待线程结束
      pthread_join(t->thread, NULL);
      t->status = T_DEAD;
    }
  }
}

// 程序退出时执行
__attribute__((destructor)) void cleanup() {
  join();
}