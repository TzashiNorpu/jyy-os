CFLAGS = -ggdb -Wall

.PHONY:a.out debug clean

run: a.out
	./a.out
a.out: stack-probe.c
	gcc $(CFLAGS) stack-probe.c

debug: a.out
	gdb -x init.gdb

clean:
	rm -rf a.out