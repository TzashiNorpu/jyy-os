T = 3

def Tsum():
  for _ in range(3):
    tmp = heap.x
    tmp +=1
    sys_sched()
    heap.x = tmp
    sys_sched()
  heap.done += 1

def main():
  heap.x = 0
  heap.done = 0
  
  for _ in range(T):
    sys_spawn(Tsum)
  
  while heap.done != T:
    sys_sched()
  
  sys_write(f'SUM = {heap.x}')