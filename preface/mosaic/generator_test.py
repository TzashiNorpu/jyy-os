#!/usr/bin/env python3
def simpleGeneratorFun():
    yield 1           
    yield 2           
    yield 3   

if __name__ == '__main__':
    ## 第一次的调用是对状态机的启动
    for value in simpleGeneratorFun():
      print(value)


  