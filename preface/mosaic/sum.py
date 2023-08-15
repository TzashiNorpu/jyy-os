T = 1

def Tsum():
  for _ in range(T):
    # 模拟读写非原子【并发模拟】
    # read
    tmp = heap.x
    tmp +=1
    # schedule
    # print("I\'m Tsum")
    sys_sched()
    # write
    heap.x = tmp
    sys_sched()
  heap.done += 1

def main():
  # print("I\'m main")
  heap.x = 0
  heap.done = 0
  
  for _ in range(T):
    sys_spawn(Tsum)
  
  while heap.done != T:
    # print("Hello")
    sys_sched()
  
  sys_write(f'SUM = {heap.x}')