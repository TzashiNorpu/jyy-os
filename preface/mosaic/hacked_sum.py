T = 3

def Tsum():
    for _ in range(T):
        tmp = heap.x
        tmp += 1
        yield ('sys_sched', ())
        heap.x = tmp
        yield ('sys_sched', ())
    heap.done += 1

def main():
    heap.x = 0
    heap.done = 0
    for _ in range(T):
        yield ('sys_spawn', (Tsum,))
    while heap.done != T:
        yield ('sys_sched', ())
    yield ('sys_write', (f'SUM = {heap.x}',))